#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on pre-rendered prompt/completion pairs.

Tokenisation is deliberately hand-rolled: the jsonl already holds the exact
strings templates/gemma3.jinja produces, so nothing here may add a second BOS,
re-apply a chat template, or move the terminator.

Batching is by token budget, not by row count. Gemma-3's vocabulary is 262k, so
the logits tensor (tokens x 262144, materialised twice: bf16 + the fp32 upcast
inside the loss) is the memory bottleneck, and a fixed row count would OOM on the
long 10-shot rows while wasting the GPU on the short ones.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


class PromptCompletionDS(Dataset):
    def __init__(self, path, tok, max_seq_len, limit=None):
        self.rows = []
        n_drop = n_total = 0
        with open(path) as f:
            for line in f:
                if limit is not None and n_total >= limit:
                    break
                r = json.loads(line)
                n_total += 1
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                if len(p) + len(c) > max_seq_len:
                    n_drop += 1
                    continue
                self.rows.append((p, c))
        self.lengths = [len(p) + len(c) for p, c in self.rows]
        s = sorted(self.lengths)
        print(
            f"[data] {path}: kept {len(self.rows)}/{n_total}, dropped-over-{max_seq_len} "
            f"{n_drop} ({n_drop / max(1, n_total):.2%}); tokens p50={s[len(s) // 2]} "
            f"p99={s[int(len(s) * 0.99)]} max={s[-1]} total={sum(s) / 1e6:.1f}M",
            flush=True,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        return {"input_ids": p + c, "labels": [-100] * len(p) + list(c)}


class TokenBudgetBatches(torch.utils.data.Sampler):
    """Length-grouped batches capped at `budget` padded tokens each."""

    def __init__(self, lengths, budget, seed=0, chunk=2048):
        self.lengths = lengths
        self.budget = budget
        self.seed = seed
        self.chunk = chunk
        self.epoch = 0
        self._n = len(self._build(0))

    def _build(self, epoch):
        rng = random.Random(self.seed + epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        batches = []
        for s in range(0, len(idx), self.chunk):
            block = sorted(idx[s:s + self.chunk], key=lambda i: self.lengths[i])
            cur, cur_max = [], 0
            for i in block:
                m = max(cur_max, self.lengths[i])
                if cur and m * (len(cur) + 1) > self.budget:
                    batches.append(cur)
                    cur, cur_max = [i], self.lengths[i]
                else:
                    cur, cur_max = cur + [i], m
            if cur:
                batches.append(cur)
        rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter(self._build(self.epoch))

    def __len__(self):
        return self._n


def collate(features, pad_id):
    m = max(len(f["input_ids"]) for f in features)
    ii, ll, aa = [], [], []
    for f in features:
        n = m - len(f["input_ids"])
        ii.append(f["input_ids"] + [pad_id] * n)
        ll.append(f["labels"] + [-100] * n)
        aa.append([1] * len(f["input_ids"]) + [0] * n)
    return {
        "input_ids": torch.tensor(ii, dtype=torch.long),
        "labels": torch.tensor(ll, dtype=torch.long),
        "attention_mask": torch.tensor(aa, dtype=torch.long),
    }


class BudgetTrainer(Trainer):
    def __init__(self, *a, batch_sampler=None, collate_fn=None, workers=4, **kw):
        super().__init__(*a, **kw)
        self._batch_sampler = batch_sampler
        self._collate_fn = collate_fn
        self._workers = workers

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=self._batch_sampler,
            collate_fn=self._collate_fn,
            num_workers=self._workers,
            pin_memory=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--token-budget", type=int, default=6144)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--param-dtype", default="float32")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = PromptCompletionDS(args.data, tok, args.max_seq_len, args.limit)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=getattr(torch, args.param_dtype),
        attn_implementation=args.attn,
    )
    # text-only task; the vision stack rides along unchanged so the checkpoint
    # stays a drop-in Gemma3ForConditionalGeneration for the grader's vLLM
    frozen = 0
    for name in ("vision_tower", "multi_modal_projector"):
        mod = getattr(model.model, name, None) if hasattr(model, "model") else None
        mod = mod if mod is not None else getattr(model, name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad_(False)
                frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {trainable / 1e9:.2f}B  frozen {frozen / 1e9:.2f}B", flush=True)
    model.config.use_cache = False
    # A parent checkpoint of ours ships a greedy generation_config (do_sample=False
    # with temperature 0 / top_k -1). transformers validates that on every
    # save_pretrained and raises, which killed a run at its first checkpoint. Carry
    # a valid config through training; the greedy one is written to disk at the end.
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        do_sample=True, top_p=0.95, top_k=64, cache_implementation="hybrid")

    sampler = TokenBudgetBatches(ds.lengths, args.token_budget, seed=args.seed)
    print(f"[batch] {len(sampler)} micro-batches/epoch, budget {args.token_budget} tok, "
          f"grad_accum {args.grad_accum} -> {len(sampler) // args.grad_accum} optimizer steps/epoch",
          flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,   # unused: batch_sampler drives the loader
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        max_grad_norm=1.0,
        optim=args.optim,
        accelerator_config={"dispatch_batches": False, "split_batches": False},
    )

    trainer = BudgetTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        batch_sampler=sampler,
        collate_fn=lambda f: collate(f, tok.pad_token_id),
    )
    trainer.train()
    print(f"[mem] peak allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
          f"reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB", flush=True)

    final = os.path.join(args.out, "final")
    model.to(torch.bfloat16)
    model.config.torch_dtype = "bfloat16"
    model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.model, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    # the graded artifact decodes with whatever generation_config.json says:
    # vLLM logs "Default sampling parameters ... overridden by the model's HF
    # generation config", and evaluate.py never sends a temperature.
    with open(os.path.join(final, "generation_config.json"), "w") as f:
        json.dump({
            "bos_token_id": 2,
            "eos_token_id": [1, 106],
            "pad_token_id": 0,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "cache_implementation": "hybrid",
        }, f, indent=2)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
