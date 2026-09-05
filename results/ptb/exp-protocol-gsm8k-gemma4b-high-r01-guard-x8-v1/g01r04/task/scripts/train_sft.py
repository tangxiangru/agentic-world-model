#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on prompt/target pairs rendered with the
grader's own chat template.

Loss is on target tokens only, and only target positions are ever pushed
through the 262k-row lm_head (chunked + rematerialised), which is what keeps a
4B model with a huge vocab inside one H100. Micro-batches are formed under a
token budget rather than a fixed row count, so short GSM8K rows still fill the
GPU; each optimizer step normalises by its own loss-token count so the gradient
does not depend on how rows happened to bucket.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_examples(path: str, max_seq_len: int, limit: int | None):
    tok = fmt.tokenizer()
    rows = [json.loads(l) for l in open(path)]
    if limit:
        rows = rows[:limit]
    sysmsg = fmt.fewshot_system() if any(
        r.get("system_mode") == "fewshot" for r in rows) else None

    prompts = [
        fmt.render(r["question"], None,
                   system=sysmsg if r.get("system_mode") == "fewshot" else None)
        for r in rows
    ]
    targets = [r["completion"] for r in rows]

    p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
    t_ids = tok(targets, add_special_tokens=False)["input_ids"]

    examples, dropped, over = [], 0, 0
    for pi, ti in zip(p_ids, t_ids):
        if len(ti) < 2:
            dropped += 1
            continue
        if len(pi) + len(ti) > max_seq_len:
            over += 1
            continue
        examples.append((pi + ti, [-100] * len(pi) + ti))
    log(f"examples={len(examples)} dropped={dropped} "
        f"over_max_seq_len={over} ({over / max(1, len(rows)):.3%})")
    lens = sorted(len(e[0]) for e in examples)
    log(f"len p50={lens[len(lens) // 2]} p95={lens[int(len(lens) * .95)]} "
        f"max={lens[-1]} total={sum(lens) / 1e6:.1f}M tokens")
    return examples


def make_microbatches(examples, token_budget: int, max_rows: int, seed: int):
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    mbs, cur, cur_max = [], [], 0
    for i in order:
        n = len(examples[i][0])
        nm = max(cur_max, n)
        if cur and (nm * (len(cur) + 1) > token_budget or len(cur) + 1 > max_rows):
            mbs.append(cur)
            cur, cur_max = [i], n
        else:
            cur.append(i)
            cur_max = nm
    if cur:
        mbs.append(cur)
    random.Random(seed).shuffle(mbs)
    return mbs


def collate(examples, idxs, pad_id, device):
    rows = [examples[i] for i in idxs]
    n = max(len(r[0]) for r in rows)
    input_ids = torch.full((len(rows), n), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), n), -100, dtype=torch.long)
    attn = torch.zeros((len(rows), n), dtype=torch.long)
    for j, (ids, lab) in enumerate(rows):
        input_ids[j, : len(ids)] = torch.tensor(ids)
        labels[j, : len(lab)] = torch.tensor(lab)
        attn[j, : len(ids)] = 1
    return (input_ids.to(device), labels.to(device), attn.to(device))


def sum_loss(model, h_sel, t_sel, chunk: int):
    """Cross-entropy summed over selected positions, lm_head rematerialised in
    chunks so full-vocab logits never all exist at once."""
    def one(h, t):
        return F.cross_entropy(model.lm_head(h).float(), t, reduction="sum")

    total = None
    for i in range(0, h_sel.size(0), chunk):
        part = torch.utils.checkpoint.checkpoint(
            one, h_sel[i: i + chunk], t_sel[i: i + chunk], use_reentrant=False
        )
        total = part if total is None else total + part
    return total


def save(model, tok, init_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(init_path, fn)
        dst = os.path.join(out_dir, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    model.config.use_cache = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--init", default=fmt.BASE_SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--max-rows", type=int, default=48)
    ap.add_argument("--loss-chunk", type=int, default=2048)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-frac", type=float, default=None,
                    help="also save a checkpoint at this fraction of training")
    args = ap.parse_args()

    log(f"args: {vars(args)}")
    log(f"chat template sha: {fmt.template_sha()}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    from transformers import Gemma3ForConditionalGeneration

    tok = fmt.tokenizer()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    examples = build_examples(args.data, args.max_seq_len, args.limit)
    label_counts = [sum(1 for x in e[1] if x != -100) for e in examples]
    mbs = make_microbatches(examples, args.token_budget, args.max_rows, args.seed)
    total_mb = int(len(mbs) * args.epochs)
    total_steps = max(1, total_mb // args.accum)
    log(f"microbatches/epoch={len(mbs)} total_mb={total_mb} opt_steps={total_steps}")

    device = "cuda"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()

    frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            frozen += p.numel()
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable {sum(p.numel() for p in trainable) / 1e9:.3f}B  "
        f"frozen {frozen / 1e6:.0f}M")

    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(trainable, lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                              weight_decay=args.weight_decay)

    warmup_steps = max(1, int(total_steps * args.warmup))

    def lr_at(s):
        if s < warmup_steps:
            return args.lr * (s + 1) / warmup_steps
        prog = min(1.0, (s - warmup_steps) / max(1, total_steps - warmup_steps))
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * cos)

    def stream():
        e = 0
        while True:
            order = mbs if e == 0 else make_microbatches(
                examples, args.token_budget, args.max_rows, args.seed + e)
            for mb in order:
                yield mb
            e += 1

    it = stream()
    save_step = int(total_steps * args.save_frac) if args.save_frac else -1
    t0 = time.time()
    seen, run, run_n = 0, 0.0, 0

    for step in range(total_steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = [next(it) for _ in range(args.accum)]
        denom = sum(label_counts[i] for mb in batch for i in mb)
        step_loss = 0.0
        for mb in batch:
            input_ids, labels, attn = collate(examples, mb, pad_id, device)
            h = model.model(input_ids=input_ids, attention_mask=attn).last_hidden_state
            h = h[:, :-1, :]
            tgt = labels[:, 1:]
            m = tgt != -100
            loss = sum_loss(model, h[m], tgt[m], args.loss_chunk) / denom
            loss.backward()
            step_loss += loss.item()
            seen += int(attn.sum().item())
            del h, loss
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        run += step_loss
        run_n += 1
        if (step + 1) % args.log_every == 0:
            el = time.time() - t0
            log(f"step {step + 1}/{total_steps} loss {run / run_n:.4f} lr {lr:.2e} "
                f"tok {seen / 1e6:.1f}M {seen / el / 1e3:.1f}k tok/s "
                f"elapsed {el / 60:.1f}m eta {el / (step + 1) * (total_steps - step - 1) / 60:.1f}m "
                f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
            run, run_n = 0.0, 0
        if step + 1 == save_step:
            d = os.path.join(args.out, f"checkpoint-{step + 1}")
            log(f"saving {d}")
            save(model, tok, args.init, d)

    log("training done, saving final")
    save(model, tok, args.init, os.path.join(args.out, "final"))
    log(f"total wall {(time.time() - t0) / 60:.1f}m")


if __name__ == "__main__":
    main()
