#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on a prompt/completion jsonl.

Rows already carry the grader's exact rendered prompt (templates/gemma3.jinja)
and a completion that ends with <end_of_turn>, so this script does no templating
of its own: it tokenizes both halves with add_special_tokens=False, masks the
prompt out of the loss, and pads dynamically.

Batching is by token budget, not by row count: rows are sorted by length and
packed into micro-batches of at most --tok-budget padded tokens. Gemma-3's
262k-token vocabulary makes the logits tensor the memory bottleneck
(~1.6 MB per position), so a fixed row count would OOM on the long
fewshot-prefixed rows and waste the GPU on the short ones.

The vision tower and multimodal projector are frozen (GSM8K is text-only); the
checkpoint is still saved as Gemma3ForConditionalGeneration so vLLM loads it
exactly the way it loads the base snapshot.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

import fmt

IGNORE = -100


class BatchDataset(Dataset):
    """Each item is a whole pre-formed micro-batch."""

    def __init__(self, path, tok, max_len, tok_budget, max_rows_per_batch, limit=None):
        rows = []
        self.n_dropped = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                d = json.loads(line)
                p = tok(d["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(d["completion"], add_special_tokens=False)["input_ids"]
                if len(p) + len(c) > max_len:
                    self.n_dropped += 1
                    continue
                rows.append((p, c))
        self.rows = rows
        self.lens = sorted(len(p) + len(c) for p, c in rows)
        self.n_tokens = sum(self.lens)

        order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]) + len(rows[i][1]))
        self.batches = []
        cur, cur_max = [], 0
        for i in order:
            L = len(rows[i][0]) + len(rows[i][1])
            m = max(cur_max, L)
            if cur and (m * (len(cur) + 1) > tok_budget or len(cur) >= max_rows_per_batch):
                self.batches.append(cur)
                cur, cur_max = [i], L
            else:
                cur.append(i)
                cur_max = m
        if cur:
            self.batches.append(cur)
        self.padded_tokens = sum(
            max(len(rows[i][0]) + len(rows[i][1]) for i in b) * len(b) for b in self.batches
        )

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, k):
        idxs = self.batches[k]
        L = max(len(self.rows[i][0]) + len(self.rows[i][1]) for i in idxs)
        ids, lab, att = [], [], []
        for i in idxs:
            p, c = self.rows[i]
            seq = p + c
            n = L - len(seq)
            ids.append(seq + [0] * n)
            lab.append([IGNORE] * len(p) + c + [IGNORE] * n)
            att.append([1] * len(seq) + [0] * n)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(lab),
            "attention_mask": torch.tensor(att),
        }


def collate(feats):
    assert len(feats) == 1
    return feats[0]


def load_model(path, liger=True):
    from transformers import Gemma3ForConditionalGeneration

    m = Gemma3ForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if liger:
        # gemma-3's 262k vocab makes logits.float() the memory wall; the fused
        # linear cross-entropy never materialises the logits tensor at all.
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3

        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True, model=m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--model", default=fmt.BASE_SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--tok-budget", type=int, default=4096)
    ap.add_argument("--max-rows-per-batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--no-liger", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    ds = BatchDataset(args.data, tok, args.max_len, args.tok_budget,
                      args.max_rows_per_batch, args.limit)
    L = ds.lens
    print(f"rows={len(ds.rows)} dropped_too_long={ds.n_dropped} micro_batches={len(ds)}", flush=True)
    print(f"len p50={L[len(L)//2]} p90={L[int(len(L)*.9)]} p99={L[int(len(L)*.99)]} max={L[-1]} "
          f"tokens={ds.n_tokens/1e6:.1f}M padded={ds.padded_tokens/1e6:.1f}M", flush=True)

    model = load_model(args.model, liger=not args.no_liger)
    model.config.use_cache = False
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable={n_train/1e9:.3f}B frozen={n_frozen/1e6:.0f}M", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=3,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        adam_beta2=0.95,
        max_grad_norm=1.0,
        dataloader_num_workers=2,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.model, fn)
        if os.path.exists(src) and not os.path.exists(os.path.join(final, fn)):
            shutil.copy(src, os.path.join(final, fn))
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
