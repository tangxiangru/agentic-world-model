#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on pre-rendered prompt/completion
jsonl, with completion-only loss.

Rows are already rendered by scripts/build_data.py to the exact string the
grader conditions on (see scripts/fmt.py), so the trainer does no templating.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

IGNORE = -100


class PackedRows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            x = b["input_ids"]
            p = b["prompt_len"]
            pad = n - len(x)
            ids.append(x + [self.pad_id] * pad)
            lab = [IGNORE] * p + x[p:] + [IGNORE] * pad
            labels.append(lab)
            mask.append([1] * len(x) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def tokenize_all(path, tok, max_seq_len, limit=None):
    prompts, completions = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            d = json.loads(line)
            prompts.append(d["prompt"])
            completions.append(d["completion"])
    pe = tok(prompts, add_special_tokens=False)["input_ids"]
    ce = tok(completions, add_special_tokens=False)["input_ids"]
    rows, dropped, lens = [], 0, []
    for p, c in zip(pe, ce):
        total = len(p) + len(c)
        lens.append(total)
        if total > max_seq_len:
            dropped += 1
            continue
        rows.append({"input_ids": p + c, "prompt_len": len(p), "length": total})
    lens.sort()
    print(
        f"[data] {len(rows)} kept, {dropped} dropped (>{max_seq_len}), "
        f"p50={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} max={lens[-1]} "
        f"drop_frac={dropped/max(1,len(lens)):.4f}",
        flush=True,
    )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=1280)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--no-gc", action="store_true", help="disable gradient checkpointing")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--liger", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    print("[tmpl] gemma3.jinja sha256 =", fmt.template_sha256(), flush=True)

    rows = tokenize_all(args.data, tok, args.max_seq_len, args.limit)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation=args.attn
    )
    model.config.use_cache = False
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {trainable/1e9:.2f}B, frozen {n_frozen/1e6:.0f}M", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=not args.no_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        optim=args.optim,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        use_liger_kernel=args.liger,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=PackedRows(rows),
        data_collator=Collator(tok.pad_token_id),
        processing_class=tok,
    )
    trainer.train()
    print("[mem] peak GiB", torch.cuda.max_memory_allocated()/2**30, flush=True)

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    # save in bf16: same dtype as the base checkpoint, half the bytes, and what
    # vLLM would cast to anyway
    model.to(torch.bfloat16)
    model.config.dtype = "bfloat16"
    if hasattr(model.config, "text_config"):
        model.config.text_config.dtype = "bfloat16"
    if hasattr(model.config, "vision_config"):
        model.config.vision_config.dtype = "bfloat16"
    trainer.save_model(final)
    tok.save_pretrained(final)
    # the grader loads this dir with vLLM as a Gemma3ForConditionalGeneration:
    # carry over the processor/vision side-car files the weights do not contain
    for fn in (
        "preprocessor_config.json",
        "processor_config.json",
        "generation_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "tokenizer.model",
    ):
        src = os.path.join(args.model, fn)
        dst = os.path.join(final, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    print("[done] saved", final, flush=True)


if __name__ == "__main__":
    main()
