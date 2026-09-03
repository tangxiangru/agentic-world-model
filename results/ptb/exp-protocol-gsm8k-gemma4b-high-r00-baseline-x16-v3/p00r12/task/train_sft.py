#!/usr/bin/env python3
"""SFT for gemma-3-4b-pt on GSM8K-style data, rendered with the grader's own
chat template (templates/gemma3.jinja) so training and grading agree byte for
byte.  Completion-only loss; every target ends with <end_of_turn>.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

TEMPLATE_PATH = "templates/gemma3.jinja"
STOP = "<end_of_turn>"


def load_template() -> str:
    with open(TEMPLATE_PATH) as f:
        t = f.read()
    print("chat template sha256:", hashlib.sha256(t.encode()).hexdigest()[:16])
    return t


class SFTData(Dataset):
    def __init__(self, rows, tok, template, max_len):
        self.rows = rows
        self.tok = tok
        self.template = template
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "input_ids": r["input_ids"],
            "labels": r["labels"],
            "length": len(r["input_ids"]),
        }


def render(tok, template, system, user):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(
        msgs, chat_template=template, tokenize=False, add_generation_prompt=True
    )


def encode_rows(rows, tok, template, max_len, verbose=True):
    out = []
    n_trunc = 0
    lens = []
    for j, r in enumerate(rows):
        prompt = render(tok, template, r.get("system"), r["user"])
        target = r["target"]
        assert target.endswith(STOP), "target does not end with the grader's stop token"
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(target, add_special_tokens=False)["input_ids"]
        ids = p_ids + t_ids
        lens.append(len(ids))
        if len(ids) > max_len:
            n_trunc += 1
            continue
        labels = [-100] * len(p_ids) + list(t_ids)
        out.append({"input_ids": ids, "labels": labels})
        if verbose and j == 0:
            print("=" * 30, "EXAMPLE 0 (prompt)", "=" * 30)
            print(repr(prompt[:1200]))
            print("=" * 30, "EXAMPLE 0 (target)", "=" * 30)
            print(repr(target[-400:]))
            full = tok(prompt + target, add_special_tokens=False)["input_ids"]
            assert full == ids, "prompt/target tokenization is not concatenative"
    if verbose:
        lens.sort()
        n = len(lens)
        print(
            f"rows={n} kept={len(out)} dropped_over_{max_len}={n_trunc} "
            f"({n_trunc / max(n,1):.2%}) p50={lens[n//2]} p99={lens[int(n*0.99)]} max={lens[-1]}"
        )
        assert n_trunc / max(n, 1) < 0.02, "more than 2% of rows exceed max_seq_len"
    return out


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            pad = m - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * len(f["input_ids"]) + [0] * pad)
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
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--save-epochs", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--grad-ckpt", type=int, default=1)
    args = ap.parse_args()

    template = load_template()
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    enc = encode_rows(rows, tok, template, args.max_seq_len)
    print(f"training rows: {len(enc)}")
    if args.dry_run:
        return

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model.config.use_cache = False
    # text-only task: freeze the vision stack
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen: {n_frozen/1e6:.0f}M   trainable: {trainable/1e6:.0f}M")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy=("epoch" if args.save_epochs else ("steps" if args.save_steps else "no")),
        save_steps=args.save_steps or 10**9,
        save_total_limit=4,
        group_by_length=True,
        length_column_name="length",
        gradient_checkpointing=bool(args.grad_ckpt),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=SFTData(enc, tok, template, args.max_seq_len),
        data_collator=Collator(tok.pad_token_id or 0),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    try:
        AutoProcessor.from_pretrained(args.model).save_pretrained(final)
    except Exception as e:  # processor is not needed for text-only serving
        print("processor save skipped:", e)
    print("saved", final)


if __name__ == "__main__":
    main()
