#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on pre-tokenised, completion-masked rows.

The vision tower and the multimodal projector are frozen; the architecture and
all non-weight files of the parent snapshot are preserved so the output dir
loads with vLLM exactly like the parent does.

Two memory notes, both learned the hard way on this box:
  * gemma-3's 262k vocab makes the materialised logits the peak allocation
    (8 rows x 2.5k tokens upcast to fp32 is ~20 GB), so the fused
    linear-cross-entropy kernel from liger is used and micro-batches are built
    to a *token* budget rather than a row count.
  * the model is held in fp32 with bf16 autocast, so a 1e-5 AdamW update is not
    rounded away by bf16's 2^-8 relative precision; the optimiser states are
    8-bit to pay for it.
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
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

PAD_ID = 0
COPY_FILES = [
    "preprocessor_config.json",
    "processor_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]


class MicroBatches(Dataset):
    """Each item is one already-grouped micro-batch (token-budgeted)."""

    def __init__(self, path: str, token_budget: int, max_rows: int, seed: int):
        rows = torch.load(path, weights_only=False)
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["input_ids"]))
        batches, cur = [], []
        for i in order:
            n = len(rows[i]["input_ids"])
            if cur and ((len(cur) + 1) * n > token_budget or len(cur) + 1 > max_rows):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(batches), generator=g).tolist()
        self.rows = rows
        self.batches = [batches[i] for i in perm]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        idxs = self.batches[i]
        n = max(len(self.rows[j]["input_ids"]) for j in idxs)
        input_ids, labels, attn = [], [], []
        for j in idxs:
            r = self.rows[j]
            ids = r["input_ids"]
            lab = [-100] * r["n_prompt"] + ids[r["n_prompt"]:]
            pad = n - len(ids)
            input_ids.append(ids + [PAD_ID] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def collate(batch):
    assert len(batch) == 1
    return batch[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--max-rows-per-batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--no-liger", action="store_true")
    args = ap.parse_args()

    ds = MicroBatches(args.data, args.token_budget, args.max_rows_per_batch, args.seed)
    n_rows = len(ds.rows)
    print(f"rows: {n_rows}  micro-batches: {len(ds)}", flush=True)

    if not args.no_liger:
        from liger_kernel.transformers import monkey_patch as mp

        mp.apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("liger gemma3 patch applied", flush=True)

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    cfg = AutoConfig.from_pretrained(args.model)
    is_mm = cfg.architectures and "ConditionalGeneration" in cfg.architectures[0]
    if is_mm:
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.model, dtype=dtype, attn_implementation=args.attn
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, attn_implementation=args.attn
        )

    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_train/1e9:.2f}B  frozen {n_frozen/1e9:.2f}B", flush=True)
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=2,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
    out = trainer.train()
    print(out, flush=True)

    final = os.path.join(args.out, "final")
    os.makedirs(final, exist_ok=True)
    model.config.use_cache = True
    model.to(torch.bfloat16)
    trainer.save_model(final)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.save_pretrained(final)
    for f in COPY_FILES:
        src = os.path.join(args.model, f)
        if os.path.exists(src) and not os.path.exists(os.path.join(final, f)):
            shutil.copy(src, os.path.join(final, f))
    # greedy decoding: evaluate.py sets no temperature, so vLLM reads these
    with open(os.path.join(final, "generation_config.json"), "w") as f:
        json.dump(
            {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
             "cache_implementation": "hybrid", "do_sample": False, "temperature": 0.0,
             "top_p": 1.0, "top_k": -1},
            f, indent=2,
        )
    with open(os.path.join(final, "train_summary.json"), "w") as f:
        json.dump({"metrics": out.metrics, "args": vars(args)}, f, indent=2)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
