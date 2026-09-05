#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on pre-rendered {prompt,target} jsonl rows.

The rows already contain the exact string templates/gemma3.jinja produces for the
grader (see build_data.py), so training and grading render identically.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


class SFTRows(Dataset):
    def __init__(self, path, tok, max_seq_len, report=True):
        self.rows = []
        n_trunc = 0
        lens = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False).input_ids
                t = tok(r["target"], add_special_tokens=False).input_ids
                ids = p + t
                lens.append(len(ids))
                if len(ids) > max_seq_len:
                    n_trunc += 1
                    continue  # drop rather than truncate: a truncated target loses its stop token
                labels = [-100] * len(p) + t[:]
                self.rows.append((ids, labels))
        if report:
            lens.sort()
            print(
                f"[data] {path}: kept={len(self.rows)} dropped_too_long={n_trunc} "
                f"({n_trunc / max(1, len(lens)):.3%}) p50={lens[len(lens) // 2]} "
                f"p99={lens[int(len(lens) * 0.99)]} max={lens[-1]} max_seq_len={max_seq_len}",
                flush=True,
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = m - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--save-epochs", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--no-liger", action="store_true")
    ap.add_argument("--attn", default="flash_attention_2")
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    ds = SFTRows(args.train_file, tok, args.max_seq_len)
    if args.max_rows:
        ds.rows = ds.rows[: args.max_rows]
        print(f"[data] truncated to {len(ds.rows)} rows")

    model = AutoModelForCausalLM.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False
    # A parent saved by this script carries generation_config {do_sample: false,
    # temperature: 0.0}; transformers refuses to SAVE that combination, which kills
    # the run at the first checkpoint. Neutralise it in memory - the greedy
    # generation_config.json is written back as a plain file after training.
    gcfg = model.generation_config
    gcfg.do_sample = False
    for k in ("temperature", "top_p", "top_k"):
        if getattr(gcfg, k, None) is not None:
            setattr(gcfg, k, None)
    gcfg.validate()
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable={n_train / 1e9:.2f}B frozen(vision)={n_frozen / 1e6:.0f}M", flush=True)

    strat = "no"
    kw = {}
    if args.save_epochs:
        strat, kw = "epoch", {}
    elif args.save_steps:
        strat, kw = "steps", {"save_steps": args.save_steps}

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy=strat,
        save_total_limit=10,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        optim="adamw_torch_fused",
        use_liger_kernel=not args.no_liger,
        max_grad_norm=1.0,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=4,
        **kw,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    out = trainer.train()
    print("[train]", out.metrics, flush=True)
    final = os.path.join(args.output_dir, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # keep the grader's decode path deterministic and stopping on <end_of_turn>
    gc = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "do_sample": False,
        "temperature": 0.0,
        "cache_implementation": "hybrid",
        "transformers_version": "4.57.3",
    }
    with open(os.path.join(final, "generation_config.json"), "w") as f:
        json.dump(gc, f, indent=2)
    # Gemma3ForConditionalGeneration is multimodal: vLLM needs the processor files,
    # which save_model does not write (pitfall: final_model_not_loadable).
    import shutil

    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    print(f"[save] {final}", flush=True)


if __name__ == "__main__":
    main()
