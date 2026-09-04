#!/usr/bin/env python3
"""SFT gemma-3-4b-pt for GSM8K, in the grader's own prompt format.

Prompts are rendered with templates/gemma3.jinja - the same jinja file
evaluate.py hands to vLLM - so training and grading see identical strings
(pitfalls.yaml:template_unreachable).  Loss is on the completion only; every
completion ends with <end_of_turn> (id 106), which is in the checkpoint's
generation_config eos_token_id list, so vLLM stops there
(pitfalls.yaml:eos_mismatch).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, Gemma3ForConditionalGeneration,
                          Trainer, TrainingArguments, set_seed)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_format as EF  # noqa: E402

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EOT_ID = 106  # <end_of_turn>
PAD_ID = 0


class SFTRows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [PAD_ID] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def build_rows(tok, path, fewshot_p, max_seq_len, seed, limit=None):
    fewshot_sys = open(os.path.join(TASK_DIR, "data", "fewshot_system.txt")).read()
    rng = random.Random(seed)
    rows, dropped = [], 0
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            sys_msg = fewshot_sys if rng.random() < fewshot_p else None
            prompt = EF.render_prompt(tok, r["question"], sys_msg)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            t_ids = tok(r["target"], add_special_tokens=False)["input_ids"]
            assert t_ids[-1] == EOT_ID, "target must end with <end_of_turn>"
            ids = p_ids + t_ids
            if len(ids) > max_seq_len:
                dropped += 1
                continue
            rows.append({
                "input_ids": ids,
                "labels": [-100] * len(p_ids) + t_ids,
                "length": len(ids),
            })
    return rows, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(TASK_DIR, "data", "sft_v1.jsonl"))
    ap.add_argument("--model", default="/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2816)
    ap.add_argument("--fewshot-p", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    rows, dropped = build_rows(tok, args.data, args.fewshot_p, args.max_seq_len,
                               args.seed, args.limit)
    lens = sorted(r["length"] for r in rows)
    print(f"rows={len(rows)} dropped_too_long={dropped} "
          f"p50={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} max={lens[-1]}",
          flush=True)
    print(f"total_tokens={sum(lens)/1e6:.1f}M loss_tokens="
          f"{sum(sum(1 for x in r['labels'] if x != -100) for r in rows)/1e6:.1f}M",
          flush=True)
    if args.dry_run:
        ex = rows[0]
        print("=== example decoded ===")
        print(tok.decode(ex["input_ids"]))
        print("=== labelled part ===")
        print(tok.decode([x for x in ex["labels"] if x != -100]))
        return

    try:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
    except Exception as e:  # e.g. flash-attn not usable for this arch/build
        print(f"attn_implementation={args.attn} failed ({e}); falling back to eager",
              flush=True)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="eager")
    for p in model.model.vision_tower.parameters():
        p.requires_grad = False
    for p in model.model.multi_modal_projector.parameters():
        p.requires_grad = False
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=3,
        bf16=True,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=SFTRows(rows),
                      data_collator=collate)
    trainer.train()
    trainer.save_model(os.path.join(args.out, "final"))
    tok.save_pretrained(os.path.join(args.out, "final"))
    print("saved", os.path.join(args.out, "final"), flush=True)


if __name__ == "__main__":
    main()
