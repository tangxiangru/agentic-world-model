#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on pre-rendered prompt/completion rows.

The data file carries strings that are already rendered with templates/gemma3.jinja
(see scripts/build_sft_data.py), so nothing here re-templates anything: the trainer
tokenizes prompt+completion, masks the prompt, and trains on the completion only.
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
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

BASE = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)


class PromptCompletionDataset(Dataset):
    def __init__(self, path, tok, max_seq_len, limit=None):
        self.rows = []
        n_trunc = 0
        lens = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                d = json.loads(line)
                p = tok(d["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(d["completion"], add_special_tokens=False)["input_ids"]
                ids = p + c
                lens.append(len(ids))
                if len(ids) > max_seq_len:
                    n_trunc += 1
                    continue
                labels = [-100] * len(p) + c[:]
                self.rows.append((ids, labels))
        lens.sort()
        self.stats = {
            "n_in": len(lens),
            "n_kept": len(self.rows),
            "dropped_too_long": n_trunc,
            "p50": lens[len(lens) // 2],
            "p99": lens[int(len(lens) * 0.99)],
            "max": lens[-1],
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            n = m - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * n)
            labels.append(f["labels"] + [-100] * n)
            attn.append([1] * len(f["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2816)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--save-total-limit", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--liger", type=int, default=1)
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    ds = PromptCompletionDataset(args.data, tok, args.max_seq_len, args.limit)
    print("DATA STATS", json.dumps(ds.stats), flush=True)
    if ds.stats["dropped_too_long"] / max(1, ds.stats["n_in"]) > 0.02:
        raise SystemExit(
            f"more than 2% of rows exceed max_seq_len={args.max_seq_len}: {ds.stats}"
        )
    if args.dry_run:
        ids, labels = ds.rows[0]
        print("--- example decode (labels part) ---")
        print(repr(tok.decode([t for t in labels if t != -100])))
        return

    if args.liger:
        # fused linear cross-entropy: gemma-3's 262k vocab makes the logits tensor
        # (tokens x 262144 x bf16, upcast to fp32 in the loss) the dominant memory
        # cost of this run; liger never materialises it.
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3

        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("liger kernel applied to gemma3", flush=True)

    from transformers import Gemma3ForConditionalGeneration

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    model.config.use_cache = False
    # the vision tower is dead weight for this task; freeze whatever exists
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {n_frozen/1e6:.0f}M  trainable {n_train/1e6:.0f}M", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        group_by_length=True,
        length_column_name="length",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        save_safetensors=True,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
        processing_class=tok,   # writes the tokenizer into every checkpoint-*/ dir
    )
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # guard the eos-collapse pitfall: Trainer has been seen to rewrite
    # generation_config.eos_token_id [1, 106] down to a scalar at save time
    import json as _json
    gp = os.path.join(final, "generation_config.json")
    gc = _json.load(open(gp)) if os.path.exists(gp) else {}
    print("saved generation_config:", gc)
    if gc.get("eos_token_id") != [1, 106]:
        print("WARNING eos_token_id is", gc.get("eos_token_id"), "-- repairing to [1, 106]")
        gc["eos_token_id"] = [1, 106]
        _json.dump(gc, open(gp, "w"), indent=2)
    print("saved", final)


if __name__ == "__main__":
    main()
