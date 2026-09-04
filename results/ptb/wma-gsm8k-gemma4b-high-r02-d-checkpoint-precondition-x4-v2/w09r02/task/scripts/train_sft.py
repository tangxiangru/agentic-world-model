#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on pre-rendered GSM8K-style rows.

The jsonl rows carry `prompt` (already rendered with templates/gemma3.jinja,
including <bos> and the trailing <start_of_turn>model header) and `completion`
(the chain of thought, the 'ANSWER: <n>' line, and the '<end_of_turn>'
terminator). Nothing here re-renders a template, so training and grading see
byte-identical strings.

Batches are formed under a token budget rather than a fixed row count, so the
peak logit tensor (vocab 262144) is bounded whatever the row lengths are.
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
    set_seed,
)

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


class Rows(Dataset):
    def __init__(self, path, tok, max_seq_len, limit=None, seed=0):
        rows = [json.loads(line) for line in open(path)]
        if limit:
            random.Random(seed).shuffle(rows)
            rows = rows[:limit]
        self.examples = []
        n_over = 0
        for r in rows:
            p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            c = tok(r["completion"], add_special_tokens=False)["input_ids"]
            if len(p) + len(c) > max_seq_len:
                n_over += 1
                continue
            self.examples.append((p, c))
        print(f"[data] kept {len(self.examples)}, dropped {n_over} over max_seq_len={max_seq_len}", flush=True)
        if rows and n_over / len(rows) > 0.02:
            raise SystemExit(f"[data] {n_over}/{len(rows)} rows exceed max_seq_len (pitfall seq_len_truncation)")
        self.lengths = [len(p) + len(c) for p, c in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        p, c = self.examples[i]
        return {"input_ids": p + c, "labels": [-100] * len(p) + list(c)}


class TokenBudgetBatches:
    """Fixed set of length-homogeneous batches; only their order is reshuffled."""

    def __init__(self, lengths, budget, seed=0, mega=64):
        rng = random.Random(seed)
        idx = list(range(len(lengths)))
        rng.shuffle(idx)
        self.batches = []
        chunk = max(1, mega * max(1, budget // max(1, int(sum(lengths) / len(lengths)))))
        for s in range(0, len(idx), chunk):
            block = sorted(idx[s : s + chunk], key=lambda i: lengths[i])
            cur, cur_max = [], 0
            for i in block:
                m = max(cur_max, lengths[i])
                if cur and m * (len(cur) + 1) > budget:
                    self.batches.append(cur)
                    cur, cur_max = [i], lengths[i]
                else:
                    cur.append(i)
                    cur_max = m
            if cur:
                self.batches.append(cur)
        self.rng = random.Random(seed + 1)
        sizes = [len(b) for b in self.batches]
        print(f"[batches] {len(self.batches)} batches, rows/batch min {min(sizes)} med "
              f"{sorted(sizes)[len(sizes)//2]} max {max(sizes)}", flush=True)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        order = list(range(len(self.batches)))
        self.rng.shuffle(order)
        for i in order:
            yield self.batches[i]


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


class BudgetTrainer(Trainer):
    batch_sampler = None
    collator = None

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collator,
            num_workers=2,
            pin_memory=True,
        )


def save_final(trainer, tok, out):
    final = os.path.join(out, "final")
    # the grader loads this with vLLM at --gpu-memory-utilization 0.3 (~24 GB):
    # an fp32 4B checkpoint (17 GB) leaves no room for a KV cache, so cast first
    trainer.model.to(torch.bfloat16)
    trainer.model.config.use_cache = True
    trainer.model.config.torch_dtype = "bfloat16"
    if hasattr(trainer.model.config, "text_config"):
        trainer.model.config.text_config.torch_dtype = "bfloat16"
    trainer.save_model(final)
    tok.save_pretrained(final)
    gc_path = os.path.join(final, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
    gc.update(
        {
            "bos_token_id": 2,
            "eos_token_id": [1, 106],
            "pad_token_id": 0,
            "do_sample": False,
            "temperature": 0.0,
            "cache_implementation": "hybrid",
        }
    )
    gc.pop("top_k", None)
    gc.pop("top_p", None)
    json.dump(gc, open(gc_path, "w"), indent=2)
    for f in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(SNAPSHOT, f)
        dst = os.path.join(final, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    print("[train] saved", final, flush=True)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parent", default=SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--token-budget", type=int, default=6144)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--load-dtype", default="float32")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--save-total-limit", type=int, default=3)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ds = Rows(args.data, tok, args.max_seq_len, args.limit, args.seed)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=getattr(torch, args.load_dtype), attn_implementation=args.attn
    )
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    model.config.use_cache = False
    # A parent checkpoint saved by this script carries do_sample=False with
    # temperature=0.0, which transformers' GenerationConfig validator rejects at
    # save time; every checkpoint save would raise. Neutralise it here and write
    # the greedy config back in save_final / finalize_ckpt.
    from transformers import GenerationConfig

    model.generation_config = GenerationConfig(bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0)

    bs = TokenBudgetBatches(ds.lengths, args.token_budget, seed=args.seed)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,  # unused: batch_sampler drives the loader
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
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        max_grad_norm=1.0,
        accelerator_config={"split_batches": False, "dispatch_batches": False},
    )

    trainer = BudgetTrainer(model=model, args=targs, train_dataset=ds)
    trainer.batch_sampler = bs
    trainer.collator = Collator(tok.pad_token_id)

    out = trainer.train()
    print("[train]", out.metrics, flush=True)
    save_final(trainer, tok, args.out)


if __name__ == "__main__":
    main()
