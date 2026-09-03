#!/usr/bin/env python3
"""Supervised fine-tuning of google/gemma-3-4b-pt for GSM8K.

Custom training loop, because the 262k-token vocabulary makes the standard
`Trainer` materialise enormous fp32 logit tensors.  Here the loss is computed
only on supervised (assistant) positions, in re-materialised chunks, and
batches are formed under a padded-token budget instead of a fixed sample count.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "templates/gemma3.jinja"


def tokenize_rows(tok, rows, max_len):
    chat_template = open(TEMPLATE).read()
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    ex, skipped = [], 0
    for r in rows:
        msgs = []
        if r.get("system"):
            msgs.append({"role": "system", "content": r["system"]})
        msgs.append({"role": "user", "content": r["user"]})
        prompt = tok.apply_chat_template(
            msgs, chat_template=chat_template, tokenize=False, add_generation_prompt=True
        )
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        c = tok(r["completion"].strip(), add_special_tokens=False)["input_ids"] + [eot]
        if len(p) + len(c) > max_len:
            skipped += 1
            continue
        ex.append((p, c))
    return ex, skipped


def make_microbatches(ex, token_budget, max_bs):
    order = sorted(range(len(ex)), key=lambda i: len(ex[i][0]) + len(ex[i][1]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        n = len(ex[i][0]) + len(ex[i][1])
        m = max(cur_max, n)
        if cur and (m * (len(cur) + 1) > token_budget or len(cur) >= max_bs):
            batches.append(cur)
            cur, cur_max = [i], n
        else:
            cur.append(i)
            cur_max = m
    if cur:
        batches.append(cur)
    return batches


def collate(ex, idxs, pad_id, device):
    m = max(len(ex[i][0]) + len(ex[i][1]) for i in idxs)
    ids, labels, attn = [], [], []
    for i in idxs:
        p, c = ex[i]
        seq = p + c
        lab = [-100] * len(p) + list(c)
        n = m - len(seq)
        ids.append(seq + [pad_id] * n)
        labels.append(lab + [-100] * n)
        attn.append([1] * len(seq) + [0] * n)
    return (
        torch.tensor(ids, device=device),
        torch.tensor(labels, device=device),
        torch.tensor(attn, device=device),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--out", default="ckpt/sft_v1")
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--max-bs", type=int, default=48)
    ap.add_argument("--samples-per-step", type=int, default=64)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss-chunk", type=int, default=8192)
    ap.add_argument("--time-budget-min", type=float, default=1e9)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = "cuda"

    rows = [json.loads(l) for l in open(args.data)]
    rng.shuffle(rows)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    tok = AutoTokenizer.from_pretrained(BASE)
    t0 = time.time()
    ex, skipped = tokenize_rows(tok, rows, args.max_len)
    ntok = sum(len(p) + len(c) for p, c in ex)
    nsup = sum(len(c) for _, c in ex)
    print(
        f"[data] {len(ex)} examples ({skipped} too long), {ntok/1e6:.1f}M tokens, "
        f"{nsup/1e6:.1f}M supervised, tokenised in {time.time()-t0:.0f}s",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.init, dtype=torch.float32, attn_implementation=os.environ.get("ATTN", "eager")
    )
    model.model.vision_tower.to(torch.bfloat16)
    model.model.multi_modal_projector.to(torch.bfloat16)
    for n, p in model.named_parameters():
        if "vision_tower" in n or "multi_modal_projector" in n:
            p.requires_grad_(False)
    model.to(dev)
    model.config.use_cache = False
    model.model.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    # transformers refuses to serialise a greedy generation config that still
    # carries sampling fields; keep the in-memory one permissive and write the
    # real decoding defaults with set_gen_config.py after saving.
    for f in ("temperature", "top_k", "top_p"):
        if hasattr(model.generation_config, f):
            setattr(model.generation_config, f, None)
    model.generation_config.do_sample = False

    text = model.model.language_model
    head = model.lm_head

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] trainable params {sum(p.numel() for p in params)/1e9:.2f}B", flush=True)

    import bitsandbytes as bnb

    opt = bnb.optim.AdamW8bit(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)

    # --- build the epoch schedule -------------------------------------------------
    mbs = make_microbatches(ex, args.token_budget, args.max_bs)
    rng.shuffle(mbs)
    steps = []
    cur, n = [], 0
    for b in mbs:
        cur.append(b)
        n += len(b)
        if n >= args.samples_per_step:
            steps.append(cur)
            cur, n = [], 0
    if cur:
        steps.append(cur)
    n_epochs_int = max(1, math.ceil(args.epochs))
    total_steps = int(len(steps) * args.epochs)
    print(f"[sched] {len(mbs)} microbatches, {len(steps)} steps/epoch, {total_steps} total", flush=True)

    warmup = max(10, int(total_steps * args.warmup_frac))
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    # cosine floor
    base_lr = args.lr

    pad_id = tok.pad_token_id or 0
    step = 0
    t_start = time.time()
    running, running_n = 0.0, 0
    stop = False
    for epoch in range(n_epochs_int):
        if epoch > 0:
            rng.shuffle(steps)
        for micro in steps:
            if step >= total_steps or stop:
                break
            sup_total = sum(
                sum(len(ex[i][1]) for i in b) for b in micro
            )
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for b in micro:
                ids, labels, attn = collate(ex, b, pad_id, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = text(input_ids=ids, attention_mask=attn).last_hidden_state
                h = h[:, :-1, :]
                lab = labels[:, 1:]
                m = lab != -100
                hs = h[m]
                ls = lab[m]
                loss = hs.new_zeros((), dtype=torch.float32)
                for i in range(0, hs.size(0), args.loss_chunk):
                    hc, lc = hs[i : i + args.loss_chunk], ls[i : i + args.loss_chunk]

                    def f(hc, lc):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            lg = head(hc)
                        return F.cross_entropy(lg.float(), lc, reduction="sum")

                    loss = loss + checkpoint(f, hc, lc, use_reentrant=False)
                (loss / sup_total).backward()
                step_loss += loss.item()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            # apply cosine floor manually
            if args.min_lr_ratio > 0:
                for g in opt.param_groups:
                    g["lr"] = max(g["lr"], base_lr * args.min_lr_ratio) if step >= warmup else g["lr"]
            step += 1
            running += step_loss
            running_n += sup_total
            if step % 10 == 0:
                el = time.time() - t_start
                print(
                    f"step {step}/{total_steps} loss {running/max(1,running_n):.4f} "
                    f"lr {opt.param_groups[0]['lr']:.2e} gn {gn:.2f} "
                    f"elapsed {el/60:.1f}m eta {(total_steps-step)*el/step/60:.1f}m",
                    flush=True,
                )
                running, running_n = 0.0, 0
            if (time.time() - t_start) / 60 > args.time_budget_min:
                print("[time] budget reached, stopping early", flush=True)
                stop = True
        if step >= total_steps or stop:
            break

    os.makedirs(args.out, exist_ok=True)
    model.config.use_cache = True
    model.to(torch.bfloat16)
    try:
        model.save_pretrained(args.out, safe_serialization=True)
    except Exception as e:  # never lose a finished run to a config validation error
        print("save_pretrained failed:", e, flush=True)
        model.generation_config = None
        model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    for f in ["preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(BASE, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    print("saved to", args.out, flush=True)


if __name__ == "__main__":
    main()
