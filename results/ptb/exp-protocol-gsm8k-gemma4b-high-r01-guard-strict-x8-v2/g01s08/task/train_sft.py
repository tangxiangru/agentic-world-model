#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on GSM8K-style data.

Rows are pre-rendered (prompt = exactly what the grader's chat template
produces; completion ends with <end_of_turn>), so training and grading see the
same string. Loss is computed on completion tokens only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
PAD_ID = 0


class SFTData(Dataset):
    def __init__(self, path, tok, max_seq_len, limit=None):
        self.rows = []
        n_trunc = 0
        with open(path) as fh:
            for i, line in enumerate(fh):
                if limit and i >= limit:
                    break
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                ids = p + c
                if len(ids) > max_seq_len:
                    n_trunc += 1
                    continue
                labels = [-100] * len(p) + c[:]
                self.rows.append((ids, labels))
        print(f"loaded {len(self.rows)} rows from {path} ({n_trunc} dropped over max_seq_len)")
        self.lengths = [len(x[0]) for x in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


def collate(feats):
    n = max(len(f["input_ids"]) for f in feats)
    input_ids, labels, attn = [], [], []
    for f in feats:
        pad = n - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [PAD_ID] * pad)
        labels.append(f["labels"] + [-100] * pad)
        attn.append([1] * len(f["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_omi2.jsonl")
    ap.add_argument("--parent", default=SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ds = SFTData(args.data, tok, args.max_seq_len, args.limit)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent,
        dtype=torch.float32 if args.dtype == "fp32" else torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    # text-only data: keep the vision tower frozen (it receives no gradient anyway)
    for n, p in model.named_parameters():
        if n.startswith("model.vision_tower") or n.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        report_to=[],
        optim=args.optim,
        gradient_checkpointing=args.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=args.seed,
        adam_beta2=0.95,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    for f in ("preprocessor_config.json", "processor_config.json", "added_tokens.json"):
        src = os.path.join(SNAPSHOT, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, f))
    print("saved", final)


if __name__ == "__main__":
    main()
