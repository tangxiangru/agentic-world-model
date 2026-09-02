#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on grader-shaped GSM8K targets.

The jsonl rows carry `prompt` (already rendered with templates/gemma3.jinja by
scripts/build_data.py) and `completion` (chain of thought, an "ANSWER: n" line,
then <end_of_turn>).  Loss is taken on the completion only.  Rows that do not
fit in --max-seq-len are dropped, never truncated: a truncated row silently
removes the stop token and the answer marker, which is the exact failure the
seq_len_truncation pitfall describes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
EXTRA_FILES = [
    "preprocessor_config.json",
    "processor_config.json",
    "added_tokens.json",
    "tokenizer.model",
]


class SFTData(Dataset):
    def __init__(self, path, tok, max_len, limit=None):
        self.rows = []
        self.lengths = []
        dropped = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                if len(p) + len(c) > max_len:
                    dropped += 1
                    continue
                self.rows.append((p, c))
                self.lengths.append(len(p) + len(c))
        self.dropped = dropped

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        ids = p + c
        labels = [-100] * len(p) + list(c)
        return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            attn.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class TokenBudgetBatches:
    """Length-sorted micro-batches with a fixed token budget.

    Gemma 3's vocabulary is 262k, so the logits tensor (B x T x 262144, plus the
    float32 copy cross-entropy makes) dominates memory and scales with B*T, not
    with B.  A fixed per-device batch size therefore OOMs on the long rows and
    wastes the GPU on the short ones; a token budget keeps peak memory flat and
    the GPU full.  Batch *order* is shuffled per epoch, contents are not.
    """

    def __init__(self, lengths, budget, seed=0, shuffle=True):
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = []
        cur, cur_max = [], 0
        for i in order:
            m = max(cur_max, lengths[i])
            if cur and m * (len(cur) + 1) > budget:
                self.batches.append(cur)
                cur, cur_max = [i], lengths[i]
            else:
                cur, cur_max = cur + [i], m
        if cur:
            self.batches.append(cur)
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        idx = list(range(len(self.batches)))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            idx = [idx[i] for i in torch.randperm(len(idx), generator=g).tolist()]
        self.epoch += 1
        for i in idx:
            yield self.batches[i]


class BudgetTrainer(Trainer):
    def __init__(self, *a, batch_sampler=None, **kw):
        self._batch_sampler = batch_sampler
        super().__init__(*a, **kw)

    def get_train_dataloader(self):
        return self.accelerator.prepare(
            DataLoader(
                self.train_dataset,
                batch_sampler=self._batch_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=True,
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=SNAPSHOT)
    ap.add_argument("--max-seq-len", type=int, default=2816)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--precision", choices=["mixed", "bf16"], default="mixed")
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--attn", default="flash_attention_2")
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    ds = SFTData(args.data, tok, args.max_seq_len, args.limit)
    print(f"[data] kept {len(ds)} rows, dropped {ds.dropped} over {args.max_seq_len} tokens "
          f"({ds.dropped / max(1, len(ds) + ds.dropped):.3%})", flush=True)
    print(f"[data] total tokens {sum(ds.lengths) / 1e6:.1f}M", flush=True)

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation=args.attn
    )
    # the vision stack is unused by this text-only task; freeze it so it keeps
    # the base weights exactly and costs no optimizer state
    frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {trainable / 1e9:.2f}B, frozen {frozen / 1e6:.0f}M", flush=True)
    model.config.use_cache = False
    # A greedy generation_config (do_sample False + temperature 0.0) is what the
    # grader needs but what GenerationConfig.validate() rejects on save, which
    # killed exp-06 at its first checkpoint. Restore a valid combination for the
    # duration of training; the greedy file is written by hand at the end.
    gcfg = model.generation_config
    gcfg.do_sample = True
    gcfg.temperature = None
    gcfg.top_k = 64
    gcfg.top_p = 0.95

    sampler = TokenBudgetBatches(ds.lengths, args.token_budget, seed=args.seed)
    print(f"[data] {len(sampler)} micro-batches, mean size "
          f"{len(ds) / max(1, len(sampler)):.1f} rows", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=5,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        optim=args.optim,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )
    trainer = BudgetTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
        batch_sampler=sampler,
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    model.to(torch.bfloat16)
    model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    for fn in EXTRA_FILES:
        src = os.path.join(args.model, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    gc_path = os.path.join(final, "generation_config.json")
    d = json.load(open(gc_path))
    d["do_sample"] = False
    d["temperature"] = 0.0
    d.pop("top_k", None)
    d.pop("top_p", None)
    json.dump(d, open(gc_path, "w"), indent=2)
    print("[done] saved", final, "with greedy generation_config", flush=True)


if __name__ == "__main__":
    main()
