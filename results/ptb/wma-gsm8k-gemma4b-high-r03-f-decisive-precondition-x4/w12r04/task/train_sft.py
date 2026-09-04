#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on GSM8K-style math word problems.

Rows are pre-rendered (prompt, target) pairs; the prompt is masked out of the
loss. The vision tower and multimodal projector are frozen but kept in the
saved checkpoint so vLLM loads the result exactly like the base model.

Two memory tricks make a full fine-tune of the 3.9B language model fit on one
H100 alongside fp32 master weights (Gemma-3's 262 208-token vocabulary makes a
naive full-position fp32 logit tensor larger than the model itself):
  * batches are formed under a padded-token budget rather than a row count;
  * the lm_head + cross-entropy is evaluated only at supervised positions, in
    re-materialised chunks.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


class SFTData(Dataset):
    def __init__(self, path, tok, max_seq_len, report=True):
        self.rows = []
        n_trunc = 0
        lens = []
        for line in open(path):
            r = json.loads(line)
            p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            t = tok(r["target"], add_special_tokens=False)["input_ids"]
            lens.append(len(p) + len(t))
            if len(p) + len(t) > max_seq_len:
                n_trunc += 1
                continue  # drop rather than truncate: a truncated target teaches no stop token
            self.rows.append((p, t))
        self.lengths = [len(p) + len(t) for p, t in self.rows]
        if report:
            a = np.array(lens)
            print(
                f"[data] {path}: kept {len(self.rows)}/{len(lens)} "
                f"(dropped {n_trunc} over max_seq_len={max_seq_len}, "
                f"{100 * n_trunc / max(1, len(lens)):.2f}%) "
                f"len p50={int(np.percentile(a, 50))} p99={int(np.percentile(a, 99))} max={int(a.max())} "
                f"supervised-token share={sum(len(t) for _, t in self.rows) / max(1, sum(self.lengths)):.2f}",
                flush=True,
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, t = self.rows[i]
        return {"input_ids": p + t, "labels": [-100] * len(p) + list(t)}


class TokenBudgetBatches(torch.utils.data.Sampler):
    """Length-sorted batches capped at `budget` padded tokens, order shuffled."""

    def __init__(self, lengths, budget, seed=0, min_bs=1):
        self.lengths = lengths
        self.budget = budget
        self.seed = seed
        self.min_bs = min_bs
        self.epoch = 0
        self._batches = self._build(seed)

    def _build(self, seed):
        rng = random.Random(seed)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)  # break ties randomly so epochs differ
        idx.sort(key=lambda i: self.lengths[i])
        batches, cur, cur_max = [], [], 0
        for i in idx:
            m = max(cur_max, self.lengths[i])
            if cur and m * (len(cur) + 1) > self.budget:
                batches.append(cur)
                cur, cur_max = [i], self.lengths[i]
            else:
                cur.append(i)
                cur_max = m
        if cur:
            batches.append(cur)
        rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._batches = self._build(self.seed + epoch)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * k)
        labels.append(b["labels"] + [-100] * k)
        attn.append([1] * len(b["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def _chunk_ce(h, y, w):
    return F.cross_entropy(F.linear(h, w).float(), y, reduction="sum")


def masked_sum_ce(model, input_ids, attention_mask, labels, chunk=4096):
    """Sum cross-entropy over supervised positions only, in checkpointed chunks."""
    base = model.module if hasattr(model, "module") else model
    # NB: we call the inner module directly, which bypasses the autocast wrapper
    # Accelerate puts on the top-level model, so enter autocast ourselves --
    # without this the whole forward silently runs in fp32 (~8x slower).
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = base.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = hidden[:, :-1, :]
        y = labels[:, 1:]
        sel = y != -100
        hs = h[sel]
        ys = y[sel]
        w = base.lm_head.weight
        total = hs.new_zeros((), dtype=torch.float32)
        for i in range(0, hs.shape[0], chunk):
            total = total + torch.utils.checkpoint.checkpoint(
                _chunk_ce, hs[i : i + chunk], ys[i : i + chunk], w, use_reentrant=False
            )
    return total.float(), int(sel.sum())


class SFTTrainer(Trainer):
    def __init__(self, *a, token_budget=12288, loss_chunk=4096, **kw):
        super().__init__(*a, **kw)
        self.token_budget = token_budget
        self.loss_chunk = loss_chunk
        # we normalise by num_items_in_batch ourselves; stop Trainer re-dividing
        self.model_accepts_loss_kwargs = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        total, n = masked_sum_ce(
            model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"],
            chunk=self.loss_chunk,
        )
        denom = num_items_in_batch if num_items_in_batch is not None else n
        loss = total / denom
        return (loss, None) if return_outputs else loss

    def get_train_dataloader(self):
        ds = self.train_dataset
        sampler = TokenBudgetBatches(ds.lengths, self.token_budget, seed=self.args.seed)
        self._batch_sampler = sampler
        return self.accelerator.prepare(
            DataLoader(
                ds,
                batch_sampler=sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=True,
            )
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--token-budget", type=int, default=12288)
    ap.add_argument("--loss-chunk", type=int, default=4096)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-loss", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    train = SFTData(args.train_file, tok, args.max_seq_len)
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    bad = sum(1 for _, t in train.rows if t[-1] != eot)
    print(f"[data] targets not ending in <end_of_turn>: {bad}", flush=True)
    assert bad == 0

    if args.dry_run:
        p, t = train.rows[0]
        print(repr(tok.decode(p)[-300:]))
        print("--- target ---")
        print(repr(tok.decode(t)))
        sampler = TokenBudgetBatches(train.lengths, args.token_budget, seed=args.seed)
        bs = [len(b) for b in sampler]
        print(f"[dry-run] {len(bs)} batches/epoch, rows/batch min={min(bs)} "
              f"median={int(np.median(bs))} max={max(bs)}")
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.parent, dtype=torch.float32, attn_implementation=args.attn
    )
    model.config.use_cache = False
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {n_train / 1e9:.3f}B", flush=True)

    if args.verify_loss:
        model = model.cuda()
        batch = collate([train[i] for i in range(2)], tok.pad_token_id)
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = model(**batch).loss
        tot, n = masked_sum_ce(model, batch["input_ids"], batch["attention_mask"],
                               batch["labels"], chunk=512)
        print(f"[verify] built-in mean CE = {ref.item():.6f}  chunked mean CE = {(tot / n).item():.6f}  n={n}")
        return

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,  # unused: a batch_sampler supplies whole batches
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=3,
        report_to=[],
        optim=args.optim,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        accelerator_config={"dispatch_batches": False, "split_batches": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=targs,
        train_dataset=train,
        data_collator=lambda b: collate(b, tok.pad_token_id),
        token_budget=args.token_budget,
        loss_chunk=args.loss_chunk,
    )
    trainer.train()

    final = os.path.join(args.output_dir, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    import shutil

    # Trainer.save_model rewrites generation_config.json and collapses
    # eos_token_id [1, 106] to the tokenizer's single eos (1). vLLM would then
    # never stop on <end_of_turn>. Overwrite it with the base file, always.
    for fn in ("preprocessor_config.json", "processor_config.json", "generation_config.json"):
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    gc = json.load(open(os.path.join(final, "generation_config.json")))
    assert gc["eos_token_id"] == [1, 106], gc
    print(f"[done] saved {final}", flush=True)


if __name__ == "__main__":
    main()
