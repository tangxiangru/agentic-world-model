#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on pre-rendered {prompt, completion} rows.

Tokenisation is explicit (add_special_tokens=False) because the prompt string
already carries the <bos> that templates/gemma3.jinja emits, and vLLM's chat
endpoint tokenises with add_special_tokens=False too. Loss is masked over the
prompt. Rows longer than --max-seq-len are dropped, not truncated, and the drop
rate is printed and asserted below --max-drop.
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
)

IGNORE = -100


class TokenBudgetBatches:
    """Length-bucketed, variable-size batches with a constant padded-token budget.

    Padding cost of a batch is len(batch) * max_len(batch); holding that product
    under a budget keeps peak activation memory flat whether the batch is 30
    short zero-shot rows or 6 long few-shot rows. Order is reshuffled per epoch.
    """

    def __init__(self, lengths, budget_tokens, seed=0, megabatch=512):
        self.lengths = lengths
        self.budget = budget_tokens
        self.seed = seed
        self.megabatch = megabatch
        self.epoch = 0
        self._n = len(self._build(0))

    def _build(self, epoch):
        rng = random.Random(self.seed + epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        batches = []
        for s in range(0, len(idx), self.megabatch):
            chunk = sorted(idx[s:s + self.megabatch],
                           key=lambda i: self.lengths[i], reverse=True)
            cur, cur_max = [], 0
            for i in chunk:
                m = max(cur_max, self.lengths[i])
                if cur and (len(cur) + 1) * m > self.budget:
                    batches.append(cur)
                    cur, cur_max = [i], self.lengths[i]
                else:
                    cur, cur_max = cur + [i], m
            if cur:
                batches.append(cur)
        rng.shuffle(batches)
        return batches

    def __len__(self):
        return self._n

    def __iter__(self):
        batches = self._build(self.epoch)
        self.epoch += 1
        # keep the epoch length stable so Trainer's step accounting stays right
        while len(batches) < self._n:
            batches.append(batches[-1])
        yield from batches[: self._n]


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
        ids, lab, att = [], [], []
        for b in batch:
            k = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * k)
            lab.append(b["labels"] + [IGNORE] * k)
            att.append([1] * len(b["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(lab, dtype=torch.long),
            "attention_mask": torch.tensor(att, dtype=torch.long),
        }


def build_rows(path, tok, max_seq_len, max_drop, limit=None):
    kept, dropped, lens = [], 0, []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            c = tok(r["completion"], add_special_tokens=False)["input_ids"]
            if len(p) + len(c) > max_seq_len:
                dropped += 1
                continue
            kept.append({"input_ids": p + c, "labels": [IGNORE] * len(p) + c})
            lens.append(len(p) + len(c))
    lens.sort()
    rate = dropped / max(1, dropped + len(kept))
    print(f"[data] kept={len(kept)} dropped={dropped} ({rate:.3%}) "
          f"len p50={lens[len(lens)//2]} p95={lens[int(.95*len(lens))]} max={lens[-1]}")
    assert rate <= max_drop, f"truncation/drop rate {rate:.3%} > {max_drop:.3%}"
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--max-drop", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--budget-tokens", type=int, default=0,
                    help="if >0, use variable-size batches with this padded-token budget")
    ap.add_argument("--no-liger", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--attn", default="flash_attention_2")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = build_rows(args.data, tok, args.max_seq_len, args.max_drop,
                      args.limit or None)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    model.config.use_cache = False

    # freeze everything that is not the text decoder: no images in this data, so
    # the vision tower/projector get no signal and would only cost optimiser state
    n_train = n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
        else:
            n_train += p.numel()
    print(f"[model] trainable={n_train/1e9:.3f}B frozen={n_frozen/1e9:.3f}B "
          f"class={type(model).__name__}")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=3,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        max_grad_norm=1.0,
        seed=args.seed,
        group_by_length=True,
        length_column_name="length",
        report_to=[],
        dataloader_num_workers=2,
        save_safetensors=True,
        use_liger_kernel=not args.no_liger,
    )

    for r in rows:
        r["length"] = len(r["input_ids"])

    if args.budget_tokens:
        targs.group_by_length = False
        batch_sampler = TokenBudgetBatches(
            [len(r["input_ids"]) for r in rows], args.budget_tokens, seed=args.seed)
        sizes = [len(b) for b in batch_sampler._build(0)]
        print(f"[batches] {len(batch_sampler)} micro-batches/epoch, "
              f"rows per batch min={min(sizes)} p50={sorted(sizes)[len(sizes)//2]} max={max(sizes)}")
    else:
        batch_sampler = None

    class BudgetTrainer(Trainer):
        def get_train_dataloader(self):
            if batch_sampler is None:
                return super().get_train_dataloader()
            from torch.utils.data import DataLoader
            dl = DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=True,
            )
            return self.accelerator.prepare(dl)

    trainer = BudgetTrainer(
        model=model,
        args=targs,
        train_dataset=PackedRows(rows),
        data_collator=Collator(tok.pad_token_id or 0),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # pitfall final_model_not_loadable: vLLM builds a processor for
    # Gemma3ForConditionalGeneration and refuses to start without these two.
    import shutil
    for extra in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.model, extra)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(final, extra))
    print(f"[done] saved {final}")


if __name__ == "__main__":
    main()
