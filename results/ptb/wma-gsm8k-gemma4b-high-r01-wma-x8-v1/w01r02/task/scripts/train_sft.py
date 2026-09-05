#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on gsm8k-style data.

Tokenisation is done here, not by a library, so the exact string the model
trains on can be printed and compared with what vLLM will render at grading
time (pitfall: template_unreachable).

  input_ids = [bos] + tok(prompt) + tok(completion)
  labels    = [-100]*len(prompt)  + tok(completion)

`completion` already ends with <end_of_turn>, the token vLLM stops on
(generation_config.eos_token_id = [1, 106], 106 = <end_of_turn>).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fmt  # noqa: E402


class PackedRows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id: int):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", type=str, default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", type=str, default="adamw_bnb_8bit")
    ap.add_argument("--save-strategy", type=str, default="no")
    ap.add_argument("--liger", type=int, default=1)
    ap.add_argument("--attn", type=str, default="flash_attention_2")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--dry-run", action="store_true", help="tokenise + report, no training")
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    eot_id = tok.convert_tokens_to_ids(fmt.END)
    assert eot_id == 106, eot_id

    rows_in = [json.loads(l) for l in open(args.data)]
    if args.max_rows > 0:
        rows_in = rows_in[: args.max_rows]

    rows, n_trunc, lens = [], 0, []
    for r in rows_in:
        p = [tok.bos_token_id] + tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c = tok(r["completion"], add_special_tokens=False)["input_ids"]
        assert c[-1] == eot_id, "target does not end with the grading stop token"
        if len(p) + len(c) > args.max_seq_len:
            n_trunc += 1
            continue
        lens.append(len(p) + len(c))
        rows.append({"input_ids": p + c, "labels": [-100] * len(p) + c})

    lens.sort()
    print(
        f"rows {len(rows)}  dropped_too_long {n_trunc} ({n_trunc/max(1,len(rows_in)):.3%})  "
        f"p50 {lens[len(lens)//2]}  p99 {lens[int(len(lens)*0.99)]}  max {lens[-1]}  "
        f"total_tokens {sum(lens)/1e6:.1f}M",
        flush=True,
    )

    # show one fully decoded row so the training string can be eyeballed
    ex = rows[0]
    print("=" * 30, "EXAMPLE ROW", "=" * 30)
    print(tok.decode(ex["input_ids"]))
    print("-" * 20, "LOSS IS ON:", "-" * 20)
    print(tok.decode([t for t in ex["labels"] if t != -100]))
    print("=" * 73, flush=True)

    if args.dry_run:
        return

    if args.liger:
        from liger_kernel.transformers import monkey_patch as _lmp

        _lmp.apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("liger fused-linear-CE patched for gemma3", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    model.config.use_cache = False
    # text-only task: freeze the SigLIP tower so no vision gradient is computed
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_steps=(args.save_steps or 500),
        save_total_limit=4,
        optim=args.optim,
        group_by_length=True,
        max_steps=args.max_steps,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=PackedRows(rows),
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    # vLLM loads Gemma3ForConditionalGeneration through the processor; keep it alongside
    try:
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(args.model).save_pretrained(final)
    except Exception as e:  # pragma: no cover
        print("processor save skipped:", e)
    print("saved", final)


if __name__ == "__main__":
    main()
