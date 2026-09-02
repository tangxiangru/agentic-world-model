"""Full-parameter SFT of google/gemma-3-4b-pt on grader-rendered prompt/completion rows.

Design notes that matter for correctness (see skills/exp_protocol/pitfalls.yaml):
  * rows are pre-rendered by build_sft.py with grader_format.render_prompt /
    render_target, so training and grading see byte-identical strings;
  * loss is completion-only (prompt tokens are -100);
  * every completion ends with <end_of_turn> (token 106), which is in the
    snapshot's generation_config eos_token_id list, so vLLM stops there;
  * rows longer than --max-seq-len are dropped, not truncated, and the count
    is printed;
  * master weights are fp32 with bf16 autocast (pure-bf16 AdamW silently drops
    updates smaller than 2^-8 relative) and the optimiser is bitsandbytes
    8-bit AdamW so the whole thing fits an 80GB card;
  * micro-batches are built to a *token* budget, not a row count: gemma-3's
    262k vocab makes the logits tensor the memory bottleneck (a 21k-token
    micro-batch needs 22GB of fp32 logits alone), so the budget caps it.

Written as a plain loop rather than HF Trainer because accelerate's autocast
wrapper upcasts the logits to fp32 on every forward, which OOMs at this vocab.
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
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration, get_cosine_schedule_with_warmup

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def load_rows(path, tok, max_seq_len, limit=None):
    rows, n_drop, lens = [], 0, []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            p = tok(r["prompt"], add_special_tokens=True)["input_ids"]
            c = tok(r["completion"], add_special_tokens=False)["input_ids"]
            if len(p) + len(c) > max_seq_len:
                n_drop += 1
                continue
            rows.append((p, c))
            lens.append(len(p) + len(c))
            if limit and len(rows) >= limit:
                break
    s = sorted(lens)
    print(f"[data] {path}: kept {len(rows)}, dropped {n_drop} over {max_seq_len} tok "
          f"({n_drop / max(1, n_drop + len(rows)):.2%}); p50={s[len(s)//2]} "
          f"p99={s[int(len(s)*0.99)]} max={s[-1]}; total={sum(s)/1e6:.1f}M tokens", flush=True)
    return rows


def build_micro_batches(rows, budget, seed):
    """Length-bucketed micro-batches whose padded token count is <= budget."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]) + len(rows[i][1]))
    batches, cur, curmax = [], [], 0
    for i in order:
        L = len(rows[i][0]) + len(rows[i][1])
        m = max(curmax, L)
        if cur and m * (len(cur) + 1) > budget:
            batches.append(cur)
            cur, curmax = [i], L
        else:
            cur.append(i)
            curmax = m
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    return batches


def collate(rows, idxs, pad_id, device):
    n = max(len(rows[i][0]) + len(rows[i][1]) for i in idxs)
    ii, ll, am = [], [], []
    for i in idxs:
        p, c = rows[i]
        ids = p + c
        k = n - len(ids)
        ii.append(ids + [pad_id] * k)
        ll.append([-100] * len(p) + list(c) + [-100] * k)
        am.append([1] * len(ids) + [0] * k)
    t = lambda x: torch.tensor(x, dtype=torch.long, device=device)  # noqa: E731
    return t(ii), t(ll), t(am)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=SNAP)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--token-budget", type=int, default=4096, help="padded tokens per micro-batch")
    ap.add_argument("--tokens-per-step", type=int, default=131072, help="tokens per optimiser step")
    ap.add_argument("--max-seq-len", type=int, default=3328)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--save-frac", type=float, nargs="*", default=[],
                    help="extra checkpoints at these fractions of training")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(args.init)
    rows = load_rows(args.data, tok, args.max_seq_len, args.limit)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init, dtype=torch.float32, attn_implementation=args.attn).to(dev)
    model.config.use_cache = False
    if not args.no_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] trainable {sum(p.numel() for p in params)/1e9:.2f}B, frozen {n_frozen/1e9:.2f}B", flush=True)

    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)

    micro = build_micro_batches(rows, args.token_budget, args.seed)
    accum = max(1, round(args.tokens_per_step / args.token_budget))
    n_micro_total = int(len(micro) * args.epochs)
    n_steps = max(1, n_micro_total // accum)
    if args.max_steps:
        n_steps = min(n_steps, args.max_steps)
    sched = get_cosine_schedule_with_warmup(opt, int(n_steps * args.warmup), n_steps,
                                            num_cycles=0.5 * (1 - args.min_lr_ratio))
    print(f"[plan] {len(micro)} micro-batches/epoch, accum={accum}, {n_steps} optimiser steps", flush=True)

    save_at = sorted({max(1, int(n_steps * f)) for f in args.save_frac})
    os.makedirs(args.out, exist_ok=True)

    def save(tag):
        d = os.path.join(args.out, tag)
        model.config.use_cache = True
        # weights are fp32 masters; ship bf16 so the config dtype matches the
        # base snapshot and vLLM does not try to serve a 17GB fp32 checkpoint
        sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
              for k, v in model.state_dict().items()}
        model.config.torch_dtype = "bfloat16"
        if hasattr(model.config, "text_config"):
            model.config.text_config.torch_dtype = "bfloat16"
        model.save_pretrained(d, state_dict=sd, safe_serialization=True)
        del sd
        tok.save_pretrained(d)
        cfg_path = os.path.join(d, "config.json")
        cfg = json.load(open(cfg_path))
        for c in (cfg, cfg.get("text_config", {}), cfg.get("vision_config", {})):
            if isinstance(c, dict):
                for k in ("dtype", "torch_dtype"):
                    if k in c or c is cfg:
                        c[k] = "bfloat16"
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        for fn in ("preprocessor_config.json", "processor_config.json", "generation_config.json"):
            src = os.path.join(args.init, fn)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(d, fn))
        model.config.use_cache = False
        print(f"[save] {d}", flush=True)

    t0 = time.time()
    step, mi, run_loss, run_tok, seen_tok = 0, 0, 0.0, 0, 0
    model.train()
    while step < n_steps:
        opt.zero_grad(set_to_none=True)
        step_loss, step_lbl = 0.0, 0
        chunk = [micro[(mi + j) % len(micro)] for j in range(accum)]
        mi += accum
        n_lbl_total = sum(
            sum(len(rows[i][1]) for i in b) for b in chunk)
        for b in chunk:
            ii, ll, am = collate(rows, b, tok.pad_token_id, dev)
            tgt = ll[:, 1:]
            sel = tgt != -100
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # run the body, then apply the 262k-wide lm_head only on the
                # positions that actually carry loss: the full-sequence logits
                # tensor is what makes this model OOM.
                hs = model.model(input_ids=ii, attention_mask=am).last_hidden_state
                logits = model.lm_head(hs[:, :-1, :][sel])
            loss = F.cross_entropy(logits.float(), tgt[sel], reduction="sum") / max(1, n_lbl_total)
            loss.backward()
            step_loss += loss.item()
            step_lbl += int(sel.sum())
            seen_tok += int(am.sum())
            del hs, logits, loss
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        run_loss += step_loss
        run_tok += step_lbl
        if step % 10 == 0 or step == 1:
            el = time.time() - t0
            print(f"[{step}/{n_steps}] loss={run_loss/max(1,min(step,10)):.4f} gn={gn:.2f} "
                  f"lr={sched.get_last_lr()[0]:.2e} tok/s={seen_tok/el:.0f} "
                  f"elapsed={el/60:.1f}m eta={(n_steps-step)*el/step/60:.1f}m "
                  f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
            run_loss, run_tok = 0.0, 0
        if step in save_at:
            save(f"step{step}")
    save("final")
    print(f"[done] {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
