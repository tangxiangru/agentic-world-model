#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt, rendered with the grader's template.

The prompt side is rendered with templates/gemma3.jinja (the exact file
evaluate.py hands to vLLM), so the training string and the grading string are
byte-identical. Loss is on the completion only, and every completion ends with
<end_of_turn> (id 106), which is in the checkpoint's generation_config
eos_token_id, so vLLM stops there.
"""
import argparse
import json
import math
import os
import shutil

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"


def build_rows(path, tok, template, max_seq_len, limit=None):
    rows = []
    n_trunc = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            d = json.loads(line)
            msgs = d["messages"]
            prompt_text = tok.apply_chat_template(
                msgs[:-1], chat_template=template, tokenize=False,
                add_generation_prompt=True)
            completion_text = d["completion"]
            assert completion_text == msgs[-1]["content"].strip() + "<end_of_turn>"
            p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
            c_ids = tok(completion_text, add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(c_ids) > max_seq_len:
                n_trunc += 1
                continue
            rows.append((p_ids, c_ids))
    return rows, n_trunc


class SFTData(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        ids = p + c
        labels = [-100] * len(p) + list(c)
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    args = ap.parse_args()

    template = open(TEMPLATE_PATH).read()
    tok = AutoTokenizer.from_pretrained(args.model)

    rows, n_trunc = build_rows(args.data, tok, template, args.max_seq_len,
                               args.limit)
    lens = sorted(len(p) + len(c) for p, c in rows)
    print(f"rows kept {len(rows)}  dropped(too long) {n_trunc}  "
          f"p50={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} max={lens[-1]}",
          flush=True)
    print("EXAMPLE PROMPT+TARGET >>>\n" + tok.decode(rows[0][0] + rows[0][1]),
          flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_train/1e9:.2f}B  frozen {n_frozen/1e9:.2f}B", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        group_by_length=True,
        length_column_name="length",
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=SFTData(rows),
                      data_collator=Collator(tok.pad_token_id or 0))
    trainer.train()

    final = os.path.join(args.out, "final")
    os.makedirs(final, exist_ok=True)
    model.config.use_cache = True
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = True
    trainer.model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    for extra in ("preprocessor_config.json", "processor_config.json",
                  "generation_config.json"):
        src = os.path.join(args.model, extra)
        if os.path.exists(src) and not os.path.exists(os.path.join(final, extra)):
            shutil.copy(src, os.path.join(final, extra))
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
