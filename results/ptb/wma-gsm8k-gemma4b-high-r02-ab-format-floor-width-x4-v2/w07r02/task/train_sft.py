#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on a prompt/completion jsonl.

The prompt field is already rendered with the grader's templates/gemma3.jinja,
so nothing here re-templates anything: prompt and completion are tokenized with
add_special_tokens=False and concatenated, and the prompt tokens are masked out
of the loss.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
EXTRA_FILES = [
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.model",
    "added_tokens.json",
]


class PackedRows(Dataset):
    def __init__(self, path, tok, max_len, limit=None):
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
                if len(ids) > max_len:
                    n_trunc += 1
                    continue
                labels = [-100] * len(p) + c[:]
                self.rows.append((ids, labels))
        print(f"[data] {len(self.rows)} rows kept, {n_trunc} dropped for len>{max_len}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        m = (m + 7) // 8 * 8
        input_ids, labels, attn = [], [], []
        for f in feats:
            n = len(f["input_ids"])
            pad = m - n
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parent", default=SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--save-strategy", default="epoch")
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--save-total-limit", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--liger", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ds = PackedRows(args.data, tok, args.max_seq_len, args.limit)

    model = AutoModelForCausalLM.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False

    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {n_train/1e9:.2f}B, frozen {n_frozen/1e6:.0f}M")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio},
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=4,
        seed=args.seed,
        remove_unused_columns=False,
        save_safetensors=True,
        use_liger_kernel=bool(args.liger),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    for f in EXTRA_FILES:
        src = os.path.join(SNAPSHOT, f)
        if os.path.exists(src) and not os.path.exists(os.path.join(final, f)):
            shutil.copy(src, os.path.join(final, f))
    print("[done] saved", final)


if __name__ == "__main__":
    main()
