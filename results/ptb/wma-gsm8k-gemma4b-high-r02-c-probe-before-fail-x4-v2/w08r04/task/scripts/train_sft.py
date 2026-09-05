#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on pre-rendered prompt/completion jsonl.

Rows are rendered by scripts/build_data.py through the *grader's own* chat template,
so the string the trainer sees is the string vLLM will see at grading time.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoProcessor, AutoTokenizer,
                          Gemma3ForConditionalGeneration, Trainer, TrainingArguments)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")


class PCDataset(Dataset):
    def __init__(self, path, tok, max_len, limit=0):
        self.rows = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                self.rows.append(r)
                if limit and len(self.rows) >= limit:
                    break
        self.tok, self.max_len = tok, max_len
        self.lengths = []
        for r in self.rows:
            p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            c = tok(r["completion"], add_special_tokens=False)["input_ids"]
            r["_p"], r["_c"] = p, c
            self.lengths.append(min(len(p) + len(c), max_len))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids = (r["_p"] + r["_c"])[: self.max_len]
        labels = ([-100] * len(r["_p"]) + r["_c"])[: self.max_len]
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class TokenBudgetBatches:
    """Length-bucketed micro-batches with a fixed *padded token* budget.

    Keeps every micro-batch at roughly the same GPU cost regardless of row length,
    which is what lets the budget be raised until the H100 is actually busy.
    """

    def __init__(self, lengths, budget, seed, epoch_seed_offset=0):
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches, cur, cur_max = [], [], 0
        for i in order:
            m = max(cur_max, lengths[i])
            if cur and m * (len(cur) + 1) > budget:
                self.batches.append(cur)
                cur, cur_max = [i], lengths[i]
            else:
                cur, cur_max = cur + [i], m
        if cur:
            self.batches.append(cur)
        self.seed, self.epoch = seed, epoch_seed_offset

    def __iter__(self):
        import random as _r
        b = list(self.batches)
        _r.Random(self.seed + self.epoch).shuffle(b)
        self.epoch += 1
        return iter(b)

    def __len__(self):
        return len(self.batches)


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, lab, att = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            lab.append(f["labels"] + [-100] * k)
            att.append([1] * len(f["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}


class BudgetTrainer(Trainer):
    batch_sampler = None

    def get_train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(self.train_dataset, batch_sampler=self.batch_sampler,
                          collate_fn=self.data_collator, num_workers=4,
                          pin_memory=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--token-budget", type=int, default=0,
                    help="padded tokens per micro-batch; 0 = use --micro-batch instead")
    ap.add_argument("--liger", action="store_true")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    ds = PCDataset(args.data, tok, args.max_len, args.limit)
    over = sum(1 for r, L in zip(ds.rows, ds.lengths)
               if len(r["_p"]) + len(r["_c"]) > args.max_len)
    print(f"rows={len(ds)} truncated={over} ({over/max(1,len(ds)):.2%}) "
          f"p50={sorted(ds.lengths)[len(ds)//2]} max={max(ds.lengths)}", flush=True)
    assert over / max(1, len(ds)) < 0.02, "more than 2% of rows truncate"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=getattr(torch, args.dtype), attn_implementation=args.attn)
    model.config.use_cache = False
    if hasattr(model.model, "vision_tower"):
        for p in model.model.vision_tower.parameters():
            p.requires_grad_(False)
        for p in model.model.multi_modal_projector.parameters():
            p.requires_grad_(False)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=4,
        report_to=[],
        seed=args.seed,
        max_grad_norm=1.0,
        save_safetensors=True,
        use_liger_kernel=args.liger,
        average_tokens_across_devices=False,
    )
    cls = BudgetTrainer if args.token_budget else Trainer
    trainer = cls(model=model, args=targs, train_dataset=ds,
                  data_collator=Collator(tok.pad_token_id))
    if args.token_budget:
        trainer.batch_sampler = TokenBudgetBatches(ds.lengths, args.token_budget, args.seed)
        print(f"token-budget batching: {len(trainer.batch_sampler)} micro-batches, "
              f"mean rows/batch {len(ds)/len(trainer.batch_sampler):.1f}", flush=True)
    res = trainer.train()
    print("train_result", res.metrics, flush=True)

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.model.to(torch.bfloat16)
    trainer.model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    try:
        AutoProcessor.from_pretrained(BASE).save_pretrained(final)
    except Exception as e:  # processor is optional for text-only grading
        print("processor save skipped:", e, flush=True)
    for f in ("generation_config.json", "preprocessor_config.json", "processor_config.json"):
        src = os.path.join(BASE, f)
        if os.path.exists(src) and not os.path.exists(os.path.join(final, f)):
            shutil.copy(src, os.path.join(final, f))
    with open(os.path.join(final, "train_metrics.json"), "w") as fh:
        json.dump(res.metrics, fh, indent=2)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
