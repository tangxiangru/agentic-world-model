#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on prompt/completion jsonl.

The prompt field is already rendered with templates/gemma3.jinja (it starts with
<bos>), so everything is tokenized with add_special_tokens=False and the
completion ends with <end_of_turn> -- the token vLLM stops on (generation_config
eos_token_id = [1, 106]).
"""
from __future__ import annotations

import argparse
import json
import os
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


class SFTData(Dataset):
    def __init__(self, path, tok, max_len, limit=None):
        self.rows = []
        n_trunc = 0
        lens = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                if len(p) + len(c) > max_len:
                    n_trunc += 1
                    continue
                self.rows.append((p, c))
                lens.append(len(p) + len(c))
        lens.sort()
        self.stats = {
            "n": len(self.rows),
            "dropped_too_long": n_trunc,
            "p50": lens[len(lens) // 2],
            "p99": lens[int(len(lens) * 0.99)],
            "max": lens[-1],
            "tokens": sum(lens),
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        ids = p + c
        labels = [-100] * len(p) + c
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        ids, labels, mask = [], [], []
        for f in feats:
            k = m - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            mask.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SNAPSHOT)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--liger", action="store_true")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ds = SFTData(args.data, tok, args.max_seq_len, args.limit)
    print("data stats:", json.dumps(ds.stats), flush=True)
    if args.stats_only:
        return

    if args.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3

        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("liger kernels applied", flush=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.text_config.use_cache = False
    model.config.use_cache = False
    # vision path is unused by this data; freeze it so it neither trains nor drifts
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=5,
        save_steps=args.save_steps if args.save_steps else 10**9,
        save_total_limit=2,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.text_config.use_cache = True
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    for f in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(SNAPSHOT, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, f))
    print("saved", final)


if __name__ == "__main__":
    main()
