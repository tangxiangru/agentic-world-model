#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on pre-rendered prompt/completion rows.

The rows already contain the grader's exact chat rendering (bos, <start_of_turn>,
the 10-shot-style user template) so nothing here re-templates: both halves are
tokenized with add_special_tokens=False and the prompt half is masked to -100.
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

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
EXTRA_FILES = [
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",  # base one, unchanged: decoding is a separate card
]


class PromptCompletionDataset(Dataset):
    def __init__(self, path, tok, max_seq_len):
        self.rows = []
        n_trunc = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                ids = p + c
                if len(ids) > max_seq_len:
                    n_trunc += 1
                    continue
                labels = [-100] * len(p) + c[:]
                self.rows.append((ids, labels))
        self.n_trunc = n_trunc

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels}


def collate(features, pad_id):
    m = max(len(f["input_ids"]) for f in features)
    input_ids, labels, attn = [], [], []
    for f in features:
        k = m - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_id] * k)
        labels.append(f["labels"] + [-100] * k)
        attn.append([1] * len(f["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


class TokenBudgetSampler(torch.utils.data.Sampler):
    """Length-bucketed batches with a cap on total tokens, so the 262k-wide
    logit tensor of a long batch cannot blow the GPU."""

    def __init__(self, lengths, max_tokens, batch_size, seed, mega=256):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.seed = seed
        self.mega = mega * batch_size
        self.epoch = 0
        self._batches = self._build(0)

    def _build(self, epoch):
        rng = random.Random(self.seed + epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        batches = []
        for s in range(0, len(idx), self.mega):
            chunk = sorted(idx[s : s + self.mega], key=lambda i: self.lengths[i])
            cur, cur_max = [], 0
            for i in chunk:
                nm = max(cur_max, self.lengths[i])
                if cur and (len(cur) >= self.batch_size or nm * (len(cur) + 1) > self.max_tokens):
                    batches.append(cur)
                    cur, cur_max = [i], self.lengths[i]
                else:
                    cur.append(i)
                    cur_max = nm
            if cur:
                batches.append(cur)
        rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # keep the batch *contents* (and therefore len) fixed across epochs so
        # the scheduler's step count stays right; only reorder them.
        order = list(self._batches)
        random.Random(self.seed + 1000 + self.epoch).shuffle(order)
        self.epoch += 1
        return iter(order)

    def __len__(self):
        return len(self._batches)


class BucketTrainer(Trainer):
    def __init__(self, *a, bucket_sampler=None, **kw):
        super().__init__(*a, **kw)
        self.bucket_sampler = bucket_sampler

    def get_train_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self.train_dataset,
            batch_sampler=self.bucket_sampler,
            collate_fn=self.data_collator,
            num_workers=2,
            pin_memory=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens-per-batch", type=int, default=9000)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-strategy", default="epoch")
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    ds = PromptCompletionDataset(args.data, tok, args.max_seq_len)
    print(f"rows={len(ds)} dropped_over_max_seq_len={ds.n_trunc}", flush=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    for name in ("vision_tower", "multi_modal_projector"):
        mod = getattr(model, name, None) or getattr(getattr(model, "model", None), name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad_(False)
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {tr/1e9:.3f}B", flush=True)

    lengths = [len(r[0]) for r in ds.rows]
    sampler = TokenBudgetSampler(
        lengths, args.max_tokens_per_batch, args.batch_size, args.seed
    )
    steps_per_epoch = math.ceil(len(sampler) / args.grad_accum)
    print(f"batches/epoch={len(sampler)} optim steps/epoch={steps_per_epoch}", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=20,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
        dataloader_drop_last=False,
    )

    trainer = BucketTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda f: collate(f, tok.pad_token_id),
        bucket_sampler=sampler,
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    for fn in EXTRA_FILES:
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
