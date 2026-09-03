#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on pre-rendered (prompt, completion) rows.

The prompt field is already the exact string templates/gemma3.jinja produces for
the grader, so nothing here re-renders a chat template.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)


class SFTRows(Dataset):
    def __init__(self, path, tok, max_len, max_rows=None, log=print):
        self.ex = []
        n_trunc = 0
        raw = []
        with open(path) as f:
            for line in f:
                raw.append(json.loads(line))
        if max_rows:
            raw = raw[:max_rows]
        for r in raw:
            p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            c = tok(r["completion"], add_special_tokens=False)["input_ids"]
            if len(p) + len(c) > max_len:
                n_trunc += 1
                continue
            ids = p + c
            labels = [-100] * len(p) + c[:]
            self.ex.append((ids, labels))
        log(f"loaded {len(self.ex)} rows from {path}; dropped {n_trunc} over {max_len} tokens")
        lens = sorted(len(a) for a, _ in self.ex)
        log(f"len p50={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} max={lens[-1]}")

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        ids, labels = self.ex[i]
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        k = m - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * k)
        labels.append(b["labels"] + [-100] * k)
        attn.append([1] * len(b["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default=SNAPSHOT)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--attn", default="flash_attention_2")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ds = SFTRows(args.data, tok, args.max_seq_len, args.max_rows)

    cfg = AutoConfig.from_pretrained(args.parent)
    from transformers import Gemma3ForConditionalGeneration

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent,
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
        config=cfg,
    )
    # text-only training: freeze the vision stack so it cannot drift
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector"):
            p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e9:.2f}B of {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        optim="adamw_torch_fused",
        use_liger_kernel=True,
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    for f in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.parent, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    # greedy, and stop on <end_of_turn> (106) / <eos> (1); vllm serve reads this file
    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump(
            {
                "bos_token_id": 2,
                "eos_token_id": [1, 106],
                "pad_token_id": 0,
                # greedy: temperature 0 is what `vllm serve --generation-config auto`
                # reads. top_k/top_p are deliberately absent - top_k=-1 is a vLLM
                # sentinel that transformers' GenerationConfig rejects on the next save.
                "do_sample": False,
                "temperature": 0.0,
                "cache_implementation": "hybrid",
                "transformers_version": "4.50.0.dev0",
            },
            f,
            indent=2,
        )
    print("saved to", args.out)


if __name__ == "__main__":
    main()
