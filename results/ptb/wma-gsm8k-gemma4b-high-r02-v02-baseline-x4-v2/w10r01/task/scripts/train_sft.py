#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on GSM8K-style data.

Rows are pre-rendered jsonl with {"prompt": str, "target": str} already in the
grader's format (see scripts/common.py). Loss is masked to the target only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

from common import SNAPSHOT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=SNAPSHOT)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--scheduler", default="cosine")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-strategy", default="epoch", choices=["no", "epoch", "steps"])
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--optim", default="adamw_bnb_8bit")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--limit", type=int, default=0, help="use only the first N rows (smoke runs)")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--dry-run", action="store_true", help="tokenize + report, no training")
    return p.parse_args()


class JsonlSFT(Dataset):
    def __init__(self, paths, tok, max_len, seed=0, limit=0):
        self.rows = []
        n_trunc = 0
        lens = []
        for path in paths:
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    p_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                    t_ids = tok(r["target"], add_special_tokens=False)["input_ids"]
                    ids = p_ids + t_ids
                    lens.append(len(ids))
                    if len(ids) > max_len:
                        n_trunc += 1
                        continue
                    labels = [-100] * len(p_ids) + t_ids[:]
                    self.rows.append((ids, labels))
        random.Random(seed).shuffle(self.rows)
        if limit:
            self.rows = self.rows[:limit]
        lens.sort()
        self.stats = {
            "n_kept": len(self.rows),
            "n_dropped_too_long": n_trunc,
            "drop_frac": n_trunc / max(1, len(lens)),
            "p50": lens[len(lens) // 2],
            "p99": lens[int(len(lens) * 0.99)],
            "max": lens[-1],
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels}


@dataclass
class Collator:
    pad_id: int

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            attn.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    args = parse_args()
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = JsonlSFT(args.data, tok, args.max_seq_len, seed=args.seed, limit=args.limit)
    print("DATA STATS", json.dumps(ds.stats), flush=True)
    if ds.stats["drop_frac"] > 0.02:
        raise SystemExit(
            f"too many rows over max_seq_len ({ds.stats['drop_frac']:.3%}); "
            f"raise --max-seq-len above {ds.stats['max']}"
        )
    if args.dry_run:
        ex = ds[0]
        print("EXAMPLE PROMPT+TARGET:\n" + tok.decode(ex["input_ids"]))
        print("SUPERVISED PART:\n" + tok.decode([t for t in ex["labels"] if t != -100]))
        return

    # fp32 master weights + bf16 autocast: at lr 1e-5 a pure-bf16 weight would
    # swallow roughly half the Adam updates (bf16 relative eps ~4e-3 vs an
    # update/weight ratio ~1e-3). 8-bit Adam states buy back the memory.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation=args.attn
    )
    # text-only task: the vision stack is carried along unchanged so that the
    # saved checkpoint loads in vLLM exactly like the base snapshot does
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    ta = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=8,
        max_steps=args.max_steps,
        group_by_length=True,
        report_to=[],
        optim=args.optim,
        seed=args.seed,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        gradient_checkpointing=False,  # enabled manually above
    )

    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id or 0),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    # weights were carried in fp32 only to keep the Adam updates; the artifact
    # ships bf16, which is what the base snapshot is and what vLLM wants
    model.to(torch.bfloat16)
    model.config.torch_dtype = "bfloat16"
    trainer.save_model(final)
    tok.save_pretrained(final)
    write_generation_config(final)
    print("SAVED", final, flush=True)


def write_generation_config(path):
    """Greedy decoding, written as raw json.

    evaluate.py never sends a temperature, so vLLM's serving layer falls back to
    default_sampling_params read from this file; the base ships
    do_sample/top_k 64/top_p 0.95 and therefore grades at T=1.0. top_k and top_p
    are omitted rather than set to sentinels (-1 is the value that makes a later
    save_pretrained raise), and the file is written with json.dump so no
    GenerationConfig validation stands between the intent and the bytes.
    """
    gen = {
        "bos_token_id": 2,
        "cache_implementation": "hybrid",
        "do_sample": False,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "temperature": 0.0,
        "transformers_version": "4.57.3",
    }
    with open(os.path.join(path, "generation_config.json"), "w") as f:
        json.dump(gen, f, indent=2)
    return gen


if __name__ == "__main__":
    main()
