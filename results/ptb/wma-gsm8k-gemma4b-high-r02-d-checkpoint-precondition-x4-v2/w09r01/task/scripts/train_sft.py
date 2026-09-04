#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on pre-rendered GSM8K-style rows.

The data file already holds `prompt` and `completion` rendered through the
grader's own templates/gemma3.jinja (see scripts/eval_format.py), so this script
does no templating of its own: it tokenizes the two strings, masks the prompt
out of the loss, and trains. That keeps training and grading on one format.

The vision tower and the multimodal projector are frozen: this is a text-only
task and the checkpoint must stay loadable as Gemma3ForConditionalGeneration so
the grader's vLLM and evaluate.py's architecture sniffing both keep working.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


class JsonlSFT(Dataset):
    def __init__(self, path, tok, max_seq_len, limit=None):
        self.rows = []
        n_trunc = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                if len(p) + len(c) > max_seq_len:
                    n_trunc += 1
                    continue
                self.rows.append((p, c))
        self.n_trunc = n_trunc
        print(f"{path}: {len(self.rows)} rows, {n_trunc} dropped for length", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        return {"input_ids": p + c, "labels": [-100] * len(p) + c}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, labs, mask = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            labs.append(f["labels"] + [-100] * k)
            mask.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labs),
            "attention_mask": torch.tensor(mask),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--liger", type=int, default=1)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--save-limit", type=int, default=2)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    ds = JsonlSFT(args.data, tok, args.max_seq_len, args.limit)

    cfg = AutoConfig.from_pretrained(args.parent)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, config=cfg, dtype=getattr(torch, args.dtype), attn_implementation=args.attn
    )
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    model.config.use_cache = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e9:.2f}B of {sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        optim=args.optim,
        gradient_checkpointing=bool(args.grad_ckpt),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=args.save_steps if args.save_steps else 10**9,
        save_total_limit=args.save_limit,
        report_to=[],
        seed=args.seed,
        group_by_length=True,
        use_liger_kernel=bool(args.liger),
        length_column_name=None,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=Collator(tok.pad_token_id))
    trainer.train()

    final = os.path.join(args.output_dir, "final")
    model.config.use_cache = True
    trainer.model.to(torch.bfloat16).save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    # the grader loads final_model/ with vLLM from a fresh process; it needs the
    # preprocessor/processor configs the multimodal architecture declares
    import shutil

    for fn in ("preprocessor_config.json", "processor_config.json", "generation_config.json", "added_tokens.json"):
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    print(f"saved {final}", flush=True)


if __name__ == "__main__":
    main()
