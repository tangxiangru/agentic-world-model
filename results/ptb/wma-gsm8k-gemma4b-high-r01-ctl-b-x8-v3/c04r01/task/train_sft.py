#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on pre-rendered chat strings.

Input jsonl rows carry `prompt` and `completion`, both already rendered in the
grader's gemma3 chat format (see build_data.py).  Loss is taken on the
completion tokens only.  The vision tower and the multimodal projector are
frozen so the optimiser only carries the language model.
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

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

BASE = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")


class SelectiveLossTrainer(Trainer):
    """Cross-entropy on the completion tokens only, in chunks.

    Gemma-3's 262k vocabulary makes a full (B, T, V) fp32 logit tensor the
    dominant memory term (17 GB at B=16, T=1024).  Only ~40% of positions carry
    a label, so gather those hidden states first and run the lm_head over them
    in chunks; peak logit memory drops by more than an order of magnitude.
    """

    chunk = 2048

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        core = model.module.model if hasattr(model, "module") else model.model
        head = model.module.lm_head if hasattr(model, "module") else model.lm_head
        # calling core/head directly bypasses accelerate's autocast wrapper on
        # model.forward, so apply autocast here or the whole pass runs in fp32
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = core(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )[0]
            h = hidden[:, :-1, :]
            lab = labels[:, 1:]
            sel = lab != -100
            hs = h[sel]
            ls = lab[sel]
            n = ls.numel()
            total = hs.new_zeros((), dtype=torch.float32)
            for i in range(0, n, self.chunk):
                logits = head(hs[i : i + self.chunk]).float()
                total = total + torch.nn.functional.cross_entropy(
                    logits, ls[i : i + self.chunk], reduction="sum"
                )
        denom = num_items_in_batch if num_items_in_batch is not None else n
        loss = total / denom
        return (loss, None) if return_outputs else loss


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

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, labels, mask = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            mask.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def build_rows(tok, path, max_seq_len, limit=None):
    prompts, completions = [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            prompts.append(d["prompt"])
            completions.append(d["completion"])
            if limit and len(prompts) >= limit:
                break
    p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
    c_ids = tok(completions, add_special_tokens=False)["input_ids"]
    rows, dropped = [], 0
    for p, c in zip(p_ids, c_ids):
        if len(p) + len(c) > max_seq_len:
            dropped += 1
            continue
        rows.append({"input_ids": p + c, "labels": [-100] * len(p) + c})
    print(f"rows {len(rows)}  dropped_too_long {dropped} "
          f"({dropped / max(1, len(p_ids)):.3%})", flush=True)
    lens = sorted(len(r["input_ids"]) for r in rows)
    if lens:
        print(f"len p50 {lens[len(lens)//2]}  p99 {lens[int(len(lens)*0.99)]} "
              f"max {lens[-1]}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--log-steps", type=int, default=20)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    rows = build_rows(tok, args.data, args.max_seq_len, args.limit)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.float32, attn_implementation="sdpa"
    )
    model.config.text_config.use_cache = False
    # the parent may carry a greedy generation_config (do_sample=False with
    # temperature/top_k set); transformers validates strictly on save and would
    # kill the run at the first checkpoint. Replace it with a minimal valid one;
    # the greedy config for serving is written back out after training.
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0)
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params {n_tr/1e9:.2f}B", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.log_steps,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=1,
        save_only_model=True,
        group_by_length=True,
        length_column_name="length",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
    )

    trainer = SelectiveLossTrainer(
        model=model,
        args=targs,
        train_dataset=PackedRows(rows),
        data_collator=Collator(tok.pad_token_id or 0),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    os.makedirs(final, exist_ok=True)
    model.config.text_config.use_cache = True
    model.to(torch.bfloat16)
    model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    with open(os.path.join(final, "generation_config.json"), "w") as f:
        json.dump({
            "bos_token_id": 2,
            "eos_token_id": [1, 106],
            "pad_token_id": 0,
            "do_sample": False,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "cache_implementation": "hybrid",
            "transformers_version": "4.50.0.dev0",
        }, f, indent=2)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
