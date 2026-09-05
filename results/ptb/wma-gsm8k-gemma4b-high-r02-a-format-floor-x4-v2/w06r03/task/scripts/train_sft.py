#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on GSM8K-style CoT data.

Renders prompts with the *grader's* chat template (templates/gemma3.jinja) and
ends every target with <end_of_turn> (token 106), which is in the base model's
generation_config eos list, so vLLM stops there at grading time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_template  # noqa: E402


class SFTRows(Dataset):
    def __init__(self, path, tok, template, max_seq_len, limit=None):
        self.rows = []
        self.lengths = []
        n_trunc = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                r = json.loads(line)
                prompt = tok.apply_chat_template(
                    r["messages"], chat_template=template,
                    tokenize=False, add_generation_prompt=True,
                )
                p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
                c_ids = tok(r["completion"], add_special_tokens=False)["input_ids"]
                ids = p_ids + c_ids
                if len(ids) > max_seq_len:
                    n_trunc += 1
                    continue
                labels = [-100] * len(p_ids) + list(c_ids)
                self.rows.append((ids, labels))
                self.lengths.append(len(ids))
        print(f"dataset {path}: {len(self.rows)} rows, dropped {n_trunc} over {max_seq_len}", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels}


def collate(features, pad_id):
    maxlen = max(len(f["input_ids"]) for f in features)
    input_ids, labels, attn = [], [], []
    for f in features:
        n = maxlen - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_id] * n)
        labels.append(f["labels"] + [-100] * n)
        attn.append([1] * len(f["input_ids"]) + [0] * n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--no-gc", action="store_true")
    args = ap.parse_args()

    from transformers import (
        AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration,
        Trainer, TrainingArguments,
    )
    from liger_kernel.transformers import apply_liger_kernel_to_gemma3

    apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    template = load_template()

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn,
    )
    # text-only training: freeze the vision tower and the projector
    n_frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {n_frozen/1e6:.1f}M, trainable {n_train/1e6:.1f}M", flush=True)
    model.config.use_cache = False

    ds = SFTRows(args.data, tok, template, args.max_seq_len, limit=args.limit)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=5,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=4,
        gradient_checkpointing=not args.no_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        report_to=[],
        seed=args.seed,
        optim=args.optim,
        max_grad_norm=1.0,
        dataloader_num_workers=2,
        save_safetensors=True,
    )

    class LenTrainer(Trainer):
        def _get_train_sampler(self, *a, **k):
            from transformers.trainer_pt_utils import LengthGroupedSampler
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                lengths=ds.lengths,
            )

    trainer = LenTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda f: collate(f, pad_id),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    try:
        AutoProcessor.from_pretrained(args.model).save_pretrained(final)
    except Exception as e:  # pragma: no cover
        print("processor save failed:", e)
    print("saved", final)


if __name__ == "__main__":
    main()
