#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on GSM8K-style math CoT.

Prompts are rendered with templates/gemma3.jinja -- the *same* file the grader
hands to vLLM -- so the training string and the eval string are byte-identical.
Targets end with token 106 (<end_of_turn>), which is in the base model's
generation_config eos list, so vLLM stops there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

SNAP = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d",
)
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"
EOT_ID = 106  # <end_of_turn>


def build_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    raw = open(TEMPLATE, "rb").read()
    print("chat template sha256:", hashlib.sha256(raw).hexdigest())
    tok.chat_template = raw.decode()
    assert tok.convert_tokens_to_ids("<end_of_turn>") == EOT_ID
    return tok


def render_prompt(tok, system, user) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class SFTRows(Dataset):
    """Pre-formed token-budget micro-batches.

    Gemma-3's vocabulary is 262144, so the fp32 logits the loss needs cost ~1 MB
    per token: a fixed batch of 8 x 2304 asks for a 17.5 GiB allocation and OOMs
    an 80 GB H100.  Batching to a *token* budget instead of a row count keeps that
    tensor bounded regardless of how long the rows in a batch are.
    """

    def __init__(self, path, tok, max_len, budget, limit=None, seed=0):
        self.ex = []
        n_drop = 0
        raw = [json.loads(l) for l in open(path)]
        if limit:
            raw = raw[:limit]
        prompts = [render_prompt(tok, r.get("system"), r["user"]) for r in raw]
        targets = [r["target"].strip() for r in raw]
        p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
        t_ids = tok(targets, add_special_tokens=False)["input_ids"]
        rows = []
        for p, t in zip(p_ids, t_ids):
            assert t[-1] == EOT_ID, "every target must end with <end_of_turn>"
            ids = p + t
            if len(ids) > max_len:
                n_drop += 1
                continue
            labels = [-100] * len(p) + t
            rows.append((ids, labels))
        lens = sorted(len(r[0]) for r in rows)
        print(f"rows kept {len(rows)} / {len(raw)} (dropped {n_drop} over max_len={max_len})")
        print(
            "len p50", lens[len(lens) // 2], "p95", lens[int(len(lens) * 0.95)],
            "max", lens[-1], "total tokens", sum(lens),
        )
        self.n_rows = len(rows)
        self.n_tokens = sum(lens)

        order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))
        cur: list[int] = []
        for i in order:
            L = len(rows[i][0])
            if cur and (len(cur) + 1) * max(L, len(rows[cur[0]][0])) > budget:
                self.ex.append([rows[j] for j in cur])
                cur = []
            cur.append(i)
        if cur:
            self.ex.append([rows[j] for j in cur])
        random.Random(seed).shuffle(self.ex)
        bs = [len(b) for b in self.ex]
        print(
            f"micro-batches {len(self.ex)} (budget {budget} tokens), "
            f"rows/batch min {min(bs)} max {max(bs)} mean {sum(bs)/len(bs):.1f}"
        )

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate(batches, pad_id=0):
    batch = batches[0]  # dataset items are already whole micro-batches
    n = max(len(ids) for ids, _ in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        k = n - len(ids)
        input_ids.append(ids + [pad_id] * k)
        labels.append(lab + [-100] * k)
        attn.append([1] * len(ids) + [0] * k)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=SNAP)
    ap.add_argument("--max-seq-len", type=int, default=2304)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok = build_tokenizer(args.model)
    ds = SFTRows(args.data, tok, args.max_seq_len, args.batch_tokens, args.limit, args.seed)
    if args.dry_run:
        ids, labels = ds.ex[0][0]
        print("---- rendered example (decoded) ----")
        print(tok.decode(ids))
        print("---- supervised part only ----")
        print(tok.decode([i for i, l in zip(ids, labels) if l != -100]))
        return

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn
    )
    # text-only task: keep the vision tower in the checkpoint but do not train it
    frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            frozen += p.numel()
    print("frozen params:", frozen / 1e6, "M")
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,  # a "row" of the dataset IS a micro-batch
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=6,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        dataloader_num_workers=2,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        accelerator_config={"split_batches": False, "dispatch_batches": False},
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id or 0),
    )
    trainer.train()
    trainer.save_model(os.path.join(args.out, "final"))
    tok.save_pretrained(os.path.join(args.out, "final"))
    print("saved to", os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
