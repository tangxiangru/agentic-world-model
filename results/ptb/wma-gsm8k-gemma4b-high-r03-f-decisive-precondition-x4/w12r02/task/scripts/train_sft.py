#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on pre-rendered prompt/completion jsonl.

Rows are already rendered through templates/gemma3.jinja (scripts/render.py), so
this script only tokenises with add_special_tokens=False and masks the prompt.

Single H100. fp32 master weights + bf16 autocast + 8-bit Adam; liger fused
linear cross-entropy keeps the 262k-vocab logits from ever being materialised.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--parent", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--max-seq-len", type=int, default=1536)
    p.add_argument("--micro-tokens", type=int, default=8192,
                   help="padded token budget per micro-batch")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every-frac", type=float, default=0.0,
                   help="also save an intermediate checkpoint at this fraction of training")
    p.add_argument("--no-liger", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="tokenise, report length stats, render one row, then exit")
    return p.parse_args()


def build_examples(path, tok, max_seq_len, limit):
    rows = []
    n_trunc = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            pi = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            ci = tok(r["completion"], add_special_tokens=False)["input_ids"]
            ids = pi + ci
            if len(ids) > max_seq_len:
                n_trunc += 1
                continue
            labels = [-100] * len(pi) + ci[:]
            rows.append((ids, labels))
    return rows, n_trunc


class TokenBatcher:
    """Length-bucketed batches under a padded-token budget, shuffled each epoch."""

    def __init__(self, rows, micro_tokens, seed):
        self.rows = rows
        self.micro_tokens = micro_tokens
        self.rng = random.Random(seed)

    def batches(self):
        order = sorted(range(len(self.rows)), key=lambda i: len(self.rows[i][0]))
        out, cur, curmax = [], [], 0
        for i in order:
            L = len(self.rows[i][0])
            m = max(curmax, L)
            if cur and m * (len(cur) + 1) > self.micro_tokens:
                out.append(cur)
                cur, curmax = [i], L
            else:
                cur.append(i)
                curmax = m
        if cur:
            out.append(cur)
        self.rng.shuffle(out)
        return out


def collate(rows, idxs, pad_id):
    L = max(len(rows[i][0]) for i in idxs)
    ids = torch.full((len(idxs), L), pad_id, dtype=torch.long)
    lab = torch.full((len(idxs), L), -100, dtype=torch.long)
    att = torch.zeros((len(idxs), L), dtype=torch.long)
    for k, i in enumerate(idxs):
        a, b = rows[i]
        ids[k, : len(a)] = torch.tensor(a)
        lab[k, : len(b)] = torch.tensor(b)
        att[k, : len(a)] = 1
    return ids, lab, att


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.parent)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    t0 = time.time()
    rows, n_trunc = build_examples(args.data, tok, args.max_seq_len, args.limit)
    lens = sorted(len(r[0]) for r in rows)
    tgt = sum(sum(1 for x in r[1] if x != -100) for r in rows)
    print(f"[data] {len(rows)} rows kept, {n_trunc} dropped over max_seq_len "
          f"({n_trunc / max(1, len(rows) + n_trunc):.3%}); "
          f"len p50={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} max={lens[-1]}; "
          f"total tokens {sum(lens)/1e6:.1f}M, target tokens {tgt/1e6:.1f}M; "
          f"tokenise {time.time()-t0:.0f}s", flush=True)

    if args.dry_run:
        a, b = rows[0]
        print("[dry-run] last 12 target token ids:", b[-12:])
        print("[dry-run] decoded tail:", repr(tok.decode(a[-40:])))
        print("[dry-run] eos id of <end_of_turn>:", tok.convert_tokens_to_ids("<end_of_turn>"))
        n_end = sum(1 for _, l in rows if l[-1] == tok.convert_tokens_to_ids("<end_of_turn>"))
        print(f"[dry-run] rows ending in <end_of_turn>: {n_end}/{len(rows)}")
        return

    if not args.no_liger:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3
        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("[model] liger kernels applied", flush=True)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.parent, dtype=torch.float32, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    # A parent saved by this script carries generation_config {do_sample: false,
    # temperature: 0.0}. transformers validates strictly on save_pretrained and
    # rejects temperature-with-do_sample-false, so the run trains fine and then dies
    # at the save. Replace it with a valid one; save() rewrites the file afterwards.
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid")
    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        for p in model.model.vision_tower.parameters():
            p.requires_grad_(False)
        for p in model.model.multi_modal_projector.parameters():
            p.requires_grad_(False)
        print("[model] vision tower frozen", flush=True)
    model.gradient_checkpointing_enable()
    model.cuda()
    model.train()  # liger only takes the fused-linear-CE path when self.training
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params {ntr/1e9:.2f}B", flush=True)

    import bitsandbytes as bnb
    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.AdamW8bit(params, lr=args.lr, betas=(0.9, 0.95),
                              weight_decay=args.weight_decay, eps=1e-8)

    batcher = TokenBatcher(rows, args.micro_tokens, args.seed)
    per_epoch = batcher.batches()
    micro_per_epoch = len(per_epoch)
    total_micro = int(micro_per_epoch * args.epochs)
    total_steps = max(1, total_micro // args.grad_accum)
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_steps * args.warmup), total_steps)
    print(f"[train] {micro_per_epoch} micro-batches/epoch, {total_steps} optimizer steps",
          flush=True)

    os.makedirs(args.out, exist_ok=True)
    save_at = int(total_steps * args.save_every_frac) if args.save_every_frac else -1

    def save(path):
        import shutil
        os.makedirs(path, exist_ok=True)
        for cfg in (model.config, getattr(model.config, "text_config", None)):
            if cfg is not None:
                cfg.torch_dtype = "bfloat16"
                cfg.dtype = "bfloat16"
        model.save_pretrained(path, safe_serialization=True,
                              state_dict={k: v.to(torch.bfloat16)
                                          for k, v in model.state_dict().items()})
        tok.save_pretrained(path)
        # the grader loads this dir with vLLM as a gemma3 multimodal model; it needs
        # the processor files the base snapshot ships (pitfall: final_model_not_loadable)
        for fn in ("preprocessor_config.json", "processor_config.json"):
            src = os.path.join(args.parent, fn)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(path, fn))
        # greedy decode: evaluate.py passes no sampling flags, so vLLM reads this file
        with open(os.path.join(path, "generation_config.json"), "w") as f:
            # no null-valued sampling fields: vLLM reads this dict and a null would
            # be a silent fallback to sampling
            json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
                       "cache_implementation": "hybrid",
                       "do_sample": False, "temperature": 0.0}, f, indent=2)
        print(f"[save] {path}", flush=True)

    step = 0
    micro = 0
    t_start = time.time()
    loss_acc, tok_acc, nb = 0.0, 0, 0
    done = False
    epoch = 0
    while not done:
        order = per_epoch if epoch == 0 else batcher.batches()
        epoch += 1
        for idxs in order:
            ids, lab, att = collate(rows, idxs, pad_id)
            ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=att, labels=lab)
                loss = out.loss
            (loss / args.grad_accum).backward()
            ntok = int((lab != -100).sum())
            loss_acc += float(loss) * ntok
            tok_acc += ntok
            nb += 1
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    el = time.time() - t_start
                    print(f"[step {step}/{total_steps}] loss {loss_acc/max(1,tok_acc):.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} "
                          f"tok/s {tok_acc/el if step==args.log_every else 0:.0f} "
                          f"elapsed {el/60:.1f}m "
                          f"eta {el/step*(total_steps-step)/60:.1f}m "
                          f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
                    loss_acc, tok_acc = 0.0, 0
                if step == save_at:
                    save(os.path.join(args.out, f"checkpoint-{step}"))
                if step >= total_steps:
                    done = True
                    break
            if micro >= total_micro:
                done = True
                break

    save(os.path.join(args.out, "final"))
    print(f"[done] {(time.time()-t_start)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
