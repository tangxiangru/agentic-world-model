#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on GSM8K-style CoT data.

Rendering uses the grader's own chat template (templates/gemma3.jinja), read
from disk and hashed, so training and grading render byte-identical strings.
Loss is computed on the assistant turn only; every target ends with
<end_of_turn>, the token vLLM stops on (generation_config eos_token_id [1,106]).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import torch
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments)

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"


class SFTRows(Dataset):
    def __init__(self, path, tokenizer, template, max_seq_len, limit=None):
        self.rows = []
        self.max_seq_len = max_seq_len
        n_trunc = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                d = json.loads(line)
                msgs = d["messages"]
                prompt_text = tokenizer.apply_chat_template(
                    msgs[:-1], chat_template=template, tokenize=False,
                    add_generation_prompt=True)
                full_text = tokenizer.apply_chat_template(
                    msgs, chat_template=template, tokenize=False,
                    add_generation_prompt=False)
                full_text = full_text.rstrip("\n")  # last supervised token is <end_of_turn>
                assert full_text.startswith(prompt_text), (prompt_text[-80:], full_text[:80])
                p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                f_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
                if len(f_ids) > max_seq_len:
                    n_trunc += 1
                    continue
                assert f_ids[: len(p_ids)] == p_ids, "tokenisation merged at the turn boundary"
                labels = list(f_ids)
                labels[: len(p_ids)] = [-100] * len(p_ids)
                self.rows.append({"input_ids": f_ids, "labels": labels})
        self.n_trunc = n_trunc

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, labels, mask = [], [], []
        for f in feats:
            pad = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            mask.append([1] * len(f["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_v1.jsonl")
    ap.add_argument("--parent", default=SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    template = open(TEMPLATE).read()
    print("template sha256:", hashlib.sha256(template.encode()).hexdigest())

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ds = SFTRows(args.data, tok, template, args.max_seq_len, args.limit)
    print(f"rows kept {len(ds)}; dropped for >max_seq_len {ds.n_trunc}")
    lens = sorted(len(r["input_ids"]) for r in ds.rows)
    print(f"len p50={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} max={lens[-1]}")
    sup = sorted(sum(1 for x in r["labels"] if x != -100) for r in ds.rows)
    print(f"supervised tokens p50={sup[len(sup)//2]} min={sup[0]}")
    ex = ds.rows[0]
    print("--- first row decoded (labels part) ---")
    print(repr(tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])[-200:]))
    if args.dry_run:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    # text-only task: freeze the vision tower and the projector
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"frozen params: {n_frozen/1e6:.0f}M; trainable: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e9:.2f}B")
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=20,
        save_steps=args.save_steps if args.save_steps else 10**9,
        save_strategy="steps" if args.save_steps else "no",
        save_total_limit=3,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        group_by_length=True,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()

    final = os.path.join(args.out, "final")
    save_model(model, tok, final)
    print("saved", final)


def save_model(model, tok, dest):
    """Save weights + tokenizer, then patch generation_config.json on disk.

    The greedy values cannot be set on the GenerationConfig object first:
    transformers 4.57 refuses to serialise do_sample=False together with
    temperature/top_k/top_p and would leave the directory with only a
    config.json. Patching the written file avoids that validation entirely.
    """
    os.makedirs(dest, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(dest, safe_serialization=True)
    tok.save_pretrained(dest)
    subprocess.run([sys.executable, "/home/ben/task/fix_gen_config.py", dest], check=True)


if __name__ == "__main__":
    main()
