#!/usr/bin/env python3
"""Completion-only SFT for google/gemma-3-4b-pt on pre-tokenized GSM8K-format data.

Deliberately a hand-written loop rather than Trainer/SFTTrainer: the two failure
modes this task punishes hardest are (a) prompt tokens leaking into the loss and
(b) rows silently truncating, and both are easier to assert on directly.

Input is <data>.tok.pt from make_train_set.py: input_ids already carry the exact
chat-template rendering the grader uses, and every target already ends in
<end_of_turn> (token 106).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoConfig, AutoTokenizer, Gemma3ForConditionalGeneration
from transformers import get_cosine_schedule_with_warmup

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
END_OF_TURN = 106
PAD_ID = 0


def build_microbatches(input_ids, prompt_lens, token_budget, max_rows, rng):
    """Length-sorted buckets so padding waste stays low, then shuffled."""
    order = sorted(range(len(input_ids)), key=lambda i: len(input_ids[i]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(input_ids[i])
        new_max = max(cur_max, L)
        if cur and (new_max * (len(cur) + 1) > token_budget or len(cur) + 1 > max_rows):
            batches.append(cur)
            cur, cur_max = [i], L
        else:
            cur.append(i)
            cur_max = new_max
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate(idxs, input_ids, prompt_lens, device):
    L = max(len(input_ids[i]) for i in idxs)
    n = len(idxs)
    ids = torch.full((n, L), PAD_ID, dtype=torch.long)
    mask = torch.zeros((n, L), dtype=torch.long)
    labels = torch.full((n, L), -100, dtype=torch.long)
    for r, i in enumerate(idxs):
        seq = input_ids[i]
        p = prompt_lens[i]
        ids[r, : len(seq)] = torch.tensor(seq)
        mask[r, : len(seq)] = 1
        labels[r, p : len(seq)] = torch.tensor(seq[p:])
    return ids.to(device), mask.to(device), labels.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path prefix of make_train_set output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=SNAP)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--token-budget", type=int, default=6144, help="tokens per micro-batch")
    ap.add_argument("--max-rows", type=int, default=32, help="rows per micro-batch")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every-frac", type=float, default=0.0,
                    help="if >0, also save an intermediate checkpoint at this fraction of training")
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda"

    blob = torch.load(args.data + ".tok.pt")
    input_ids, prompt_lens = blob["input_ids"], blob["prompt_lens"]
    assert all(len(s) <= args.max_seq_len for s in input_ids), "row longer than max_seq_len"
    assert all(s[-1] == END_OF_TURN for s in input_ids), "a target does not end in <end_of_turn>"
    assert all(0 < p < len(s) for p, s in zip(prompt_lens, input_ids)), "bad prompt boundary"
    n_target_tokens = sum(len(s) - p for s, p in zip(input_ids, prompt_lens))
    print(f"[data] {len(input_ids)} rows, {sum(map(len,input_ids))/1e6:.2f}M tokens, "
          f"{n_target_tokens/1e6:.2f}M loss tokens", flush=True)

    tok = AutoTokenizer.from_pretrained(SNAP)
    print(f"[model] loading {args.init}", flush=True)
    try:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.init, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    except Exception as e:  # pragma: no cover
        print("[model] flash_attention_2 unavailable, falling back to sdpa:", e, flush=True)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.init, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to(device)
    # text-only task: the vision stack never sees a gradient
    n_frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {n_train/1e9:.2f}B, frozen {n_frozen/1e9:.2f}B", flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()

    epochs_int = math.ceil(args.epochs)
    all_batches = []
    for e in range(epochs_int):
        bs = build_microbatches(input_ids, prompt_lens, args.token_budget, args.max_rows,
                                random.Random(args.seed + e))
        all_batches.extend(bs)
    keep = int(len(all_batches) * (args.epochs / epochs_int))
    all_batches = all_batches[:keep]
    n_micro = len(all_batches)
    n_steps = n_micro // args.accum
    if args.max_steps > 0:
        n_steps = min(n_steps, args.max_steps)
        n_micro = n_steps * args.accum
        all_batches = all_batches[:n_micro]
    print(f"[plan] {n_micro} micro-batches, accum {args.accum} -> {n_steps} optimizer steps", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8, fused=True)
    sched = get_cosine_schedule_with_warmup(
        opt, int(args.warmup * n_steps), n_steps,
        num_cycles=0.5 * (1 - args.min_lr_frac) if args.min_lr_frac else 0.5)

    save_at = int(n_steps * args.save_every_frac) if args.save_every_frac > 0 else -1
    t0 = time.time()
    step = 0
    run_loss, run_tok = 0.0, 0
    history = []
    for mi in range(n_micro):
        idxs = all_batches[mi]
        ids, mask, labels = collate(idxs, input_ids, prompt_lens, device)
        out = model(input_ids=ids, attention_mask=mask)
        logits = out.logits[:, :-1, :]
        tgt = labels[:, 1:]
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1),
            ignore_index=-100, reduction="sum")
        ntok = (tgt != -100).sum()
        (loss_sum / ntok / args.accum).backward()
        run_loss += loss_sum.item()
        run_tok += ntok.item()
        del out, logits, loss_sum

        if (mi + 1) % args.accum == 0:
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == 1:
                el = time.time() - t0
                loss = run_loss / max(1, run_tok)
                history.append({"step": step, "loss": loss, "lr": sched.get_last_lr()[0],
                                "grad_norm": float(gn), "elapsed_s": el})
                print(f"[step {step}/{n_steps}] loss {loss:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"gnorm {float(gn):.2f} elapsed {el/60:.1f}m eta {(el/step)*(n_steps-step)/60:.1f}m",
                      flush=True)
                run_loss, run_tok = 0.0, 0
            if save_at > 0 and step == save_at:
                save(model, tok, args.out + f"-step{step}", args.init)

    save(model, tok, args.out, args.init)
    with open(args.out + "/train_history.json", "w") as f:
        json.dump({"history": history, "args": vars(args)}, f, indent=2)
    print(f"[done] {(time.time()-t0)/60:.1f} min", flush=True)


def save(model, tok, path, init):
    os.makedirs(path, exist_ok=True)
    print(f"[save] {path}", flush=True)
    model.config.use_cache = True
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    # keep the generation_config the grader's vLLM reads (eos_token_id [1, 106])
    for fn in ["generation_config.json", "preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(SNAP, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(path, fn))
    model.config.use_cache = False


if __name__ == "__main__":
    main()
