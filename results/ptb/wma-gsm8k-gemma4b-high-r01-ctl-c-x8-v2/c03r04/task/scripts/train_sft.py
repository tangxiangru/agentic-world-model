#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on pre-rendered {prompt, completion} jsonl.

The jsonl rows are already rendered with the grader's own chat template
(templates/gemma3.jinja), so the training strings are byte-identical to what
vLLM sees at grading time. Loss is on completion tokens only.

Two memory tricks, both needed to fit a 262k-vocab 4B model on one H100:
  * the LM head runs only on positions that carry a label, not the whole
    sequence (prompt tokens are ~60% of the corpus and ~90% of a few-shot row);
  * batches are built to a token budget, not a fixed row count, after sorting
    by length, so padding is small and peak activation memory is bounded.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

BASE = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
IGNORE = -100


class BatchedRows(Dataset):
    """Each item is one already-formed micro-batch (list of rows)."""

    def __init__(self, path, tok, max_len, token_budget, max_rows_per_batch,
                 limit=None, seed=0, epochs=1):
        rows, lens = [], []
        n_drop = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                lens.append(len(p) + len(c))
                if len(p) + len(c) > max_len:
                    n_drop += 1
                    continue
                rows.append((p + c, [IGNORE] * len(p) + c))
        lens.sort()
        self.stats = {
            "n_rows_kept": len(rows), "n_dropped_over_max_len": n_drop,
            "trunc_frac": round(n_drop / max(1, len(lens)), 5),
            "len_p50": lens[len(lens) // 2], "len_p99": lens[int(len(lens) * 0.99)],
            "len_max": lens[-1],
            "completion_tokens": sum(sum(1 for x in lb if x != IGNORE) for _, lb in rows),
            "total_tokens": sum(len(a) for a, _ in rows),
        }

        rng = random.Random(seed)
        self.batches = []
        for ep in range(int(epochs)):
            order = list(range(len(rows)))
            rng.shuffle(order)
            # sort into length-homogeneous chunks so padding stays small but the
            # epoch order still varies
            order.sort(key=lambda i: (len(rows[i][0]) // 64, rng.random()))
            ep_batches = []
            cur, cur_max = [], 0
            for i in order:
                L = len(rows[i][0])
                nmax = max(cur_max, L)
                if cur and (nmax * (len(cur) + 1) > token_budget
                            or len(cur) >= max_rows_per_batch):
                    ep_batches.append(cur)
                    cur, cur_max = [i], L
                else:
                    cur.append(i)
                    cur_max = nmax
            if cur:
                ep_batches.append(cur)
            rng.shuffle(ep_batches)  # shuffle within the epoch, keep epochs ordered
            self.batches.extend(ep_batches)
        self.rows = rows
        self.stats["n_batches"] = len(self.batches)
        self.stats["epochs"] = epochs

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return [self.rows[j] for j in self.batches[i]]


def make_collate(pad_id):
    def collate(batch):
        rows = batch[0]  # dataset items are already micro-batches
        n = max(len(a) for a, _ in rows)
        ids, labels, attn = [], [], []
        for a, lb in rows:
            k = n - len(a)
            ids.append(a + [pad_id] * k)
            labels.append(lb + [IGNORE] * k)
            attn.append([1] * len(a) + [0] * k)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    return collate


class SparseHeadTrainer(Trainer):
    """Cross-entropy on labelled positions only; the LM head never sees the prompt."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        base = model.model if hasattr(model, "model") else model.base_model
        out = base(input_ids=inputs["input_ids"],
                   attention_mask=inputs["attention_mask"],
                   use_cache=False)
        h = out.last_hidden_state
        shift_h = h[:, :-1, :]
        shift_y = labels[:, 1:]
        mask = shift_y != IGNORE
        sel_h = shift_h[mask]
        sel_y = shift_y[mask]
        logits = model.lm_head(sel_h).float()
        loss = F.cross_entropy(logits, sel_y, reduction="sum")
        denom = num_items_in_batch if num_items_in_batch is not None else sel_y.numel()
        if torch.is_tensor(denom):
            denom = denom.to(loss.device)
        loss = loss / denom
        return (loss, out) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--token-budget", type=int, default=10240)
    ap.add_argument("--max-rows-per-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(BASE)
    ds = BatchedRows(args.data, tok, args.max_len, args.token_budget,
                     args.max_rows_per_batch, limit=args.limit, seed=args.seed,
                     epochs=args.epochs)
    print("data stats:", json.dumps(ds.stats), flush=True)
    if args.stats_only:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation=args.attn)
    model.config.use_cache = False
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_train/1e9:.2f}B  frozen {n_frozen/1e6:.0f}M", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=1,  # epochs are materialised inside BatchedRows
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 10**9,
        save_total_limit=4,
        save_only_model=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        save_safetensors=True,
        accelerator_config={"dispatch_batches": False},
    )

    trainer = SparseHeadTrainer(model=model, args=targs, train_dataset=ds,
                                data_collator=make_collate(tok.pad_token_id))
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    for fn in ("preprocessor_config.json", "processor_config.json",
               "generation_config.json"):
        src = os.path.join(BASE, fn)
        dst = os.path.join(final, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    print("saved", final, flush=True)
    print("peak_mem_GB", torch.cuda.max_memory_allocated() / 1e9, flush=True)


if __name__ == "__main__":
    main()
