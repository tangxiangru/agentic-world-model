"""Completion-only SFT for google/gemma-3-4b-pt on GSM8K-style data.

Rows are {"prompt", "completion"} already rendered with templates/gemma3.jinja
(see scripts/render.py); the completion ends with <end_of_turn>, the token vLLM
stops on. Loss is taken on completion tokens only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)

EXTRA_FILES = [
    "preprocessor_config.json",
    "processor_config.json",
    "added_tokens.json",
    "tokenizer.model",
]


class SFTRows(Dataset):
    def __init__(self, path, tok, max_seq_len, limit=None):
        self.rows = []
        n_drop = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                if len(p) + len(c) > max_seq_len:
                    n_drop += 1
                    continue
                self.rows.append((p, c))
        self.n_drop = n_drop

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        ids = p + c
        labels = [-100] * len(p) + list(c)
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class TokenBudgetBatches:
    """Length-sorted micro-batches with a fixed token budget.

    Gemma-3's vocabulary is 262144, so the cross-entropy logits tensor costs
    ~1.6 MB per token; a fixed micro-batch of 8 long rows asks for 15.7 GiB and
    OOMs. Budgeting padded tokens per micro-batch instead keeps peak memory flat
    and lets short rows ride in large batches.
    """

    def __init__(self, lengths, budget, max_bs, seed):
        self.batches = []
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        cur, cur_max = [], 0
        for i in order:
            m = max(cur_max, lengths[i])
            if cur and (m * (len(cur) + 1) > budget or len(cur) >= max_bs):
                self.batches.append(cur)
                cur, cur_max = [i], lengths[i]
            else:
                cur.append(i)
                cur_max = m
        if cur:
            self.batches.append(cur)
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        import random as _r

        order = list(range(len(self.batches)))
        _r.Random(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        for i in order:
            yield self.batches[i]


class TokenBudgetTrainer(Trainer):
    batch_sampler = None

    def get_train_dataloader(self):
        dl = DataLoader(
            self.train_dataset,
            batch_sampler=self.batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )
        return self.accelerator.prepare(dl)


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        d = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * d)
        labels.append(b["labels"] + [-100] * d)
        attn.append([1] * len(b["input_ids"]) + [0] * d)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--max-bs", type=int, default=32)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--attn", default="flash_attention_2")
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    ds = SFTRows(args.data, tok, args.max_seq_len, args.limit_rows)
    tot = sum(len(r[0]) + len(r[1]) for r in ds.rows)
    print(
        f"[data] {len(ds)} rows kept, {ds.n_drop} dropped for exceeding "
        f"max_seq_len={args.max_seq_len}; {tot/1e6:.1f}M tokens/epoch",
        flush=True,
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=False,
        report_to=[],
        seed=args.seed,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        save_safetensors=True,
    )

    batcher = TokenBudgetBatches(
        [len(p) + len(c) for p, c in ds.rows], args.token_budget, args.max_bs, args.seed
    )
    sizes = [len(b) for b in batcher.batches]
    print(
        f"[batches] {len(batcher)} micro-batches/epoch, rows/batch "
        f"min={min(sizes)} median={sorted(sizes)[len(sizes)//2]} max={max(sizes)}; "
        f"optimizer steps/epoch ~{len(batcher)//args.accum}",
        flush=True,
    )

    trainer = TokenBudgetTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.batch_sampler = batcher
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    for fn in EXTRA_FILES:
        src = os.path.join(args.model, fn)
        if os.path.exists(src) and not os.path.exists(os.path.join(final, fn)):
            shutil.copy(src, os.path.join(final, fn))
    print(f"[done] saved to {final}", flush=True)


if __name__ == "__main__":
    main()
