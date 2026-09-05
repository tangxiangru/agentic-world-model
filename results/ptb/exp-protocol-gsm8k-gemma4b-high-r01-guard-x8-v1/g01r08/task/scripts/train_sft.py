#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on GSM8K-style data.

The prompt/target strings are rendered with the SAME chat template the grader
uses (templates/gemma3.jinja) - the script asserts its manual rendering is
byte-identical to a jinja render of that file before training starts.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

TEMPLATE_PATH = "templates/gemma3.jinja"


def render_prompt(user_text: str) -> str:
    """Exactly what templates/gemma3.jinja produces for a single user turn."""
    return (
        "<bos><start_of_turn>user\n"
        + user_text.strip()
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def verify_template(tok, sample_user: str) -> None:
    with open(TEMPLATE_PATH) as f:
        template = f.read()
    ref = tok.apply_chat_template(
        [{"role": "user", "content": sample_user}],
        chat_template=template,
        tokenize=False,
        add_generation_prompt=True,
    )
    mine = render_prompt(sample_user)
    if ref != mine:
        raise SystemExit(
            "template mismatch!\n--- jinja ---\n%r\n--- mine ---\n%r" % (ref, mine)
        )
    print("[template] manual render == templates/gemma3.jinja render  OK")


class SFTData(Dataset):
    def __init__(self, rows, tok, max_len: int):
        self.ex = []
        n_trunc = 0
        for r in rows:
            p_ids = tok(render_prompt(r["prompt"]), add_special_tokens=False)["input_ids"]
            tgt = r["target"].strip()
            assert tgt.endswith("<end_of_turn>"), tgt[-40:]
            t_ids = tok(tgt, add_special_tokens=False)["input_ids"]
            assert t_ids[-1] == 106, t_ids[-3:]
            ids = p_ids + t_ids
            if len(ids) > max_len:
                n_trunc += 1
                continue
            labels = [-100] * len(p_ids) + t_ids[:]
            self.ex.append({"input_ids": ids, "labels": labels, "length": len(ids)})
        print(f"[data] kept {len(self.ex)} rows, dropped {n_trunc} over max_len={max_len}"
              f" ({n_trunc / max(1, len(rows)):.3%})")
        assert self.ex, "no rows left"

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        m = (m + 7) // 8 * 8
        input_ids, labels, attn = [], [], []
        for f in feats:
            k = m - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            attn.append([1] * len(f["input_ids"]) + [0] * k)
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
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--attn", default="flash_attention_2")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    verify_template(tok, "hello world")

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit_rows:
        rows = rows[: args.limit_rows]
    ds = SFTData(rows, tok, args.max_seq_len)

    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    print("[model]", type(model).__name__)
    # freeze the vision stack: no image data here
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"[model] frozen vision params: {n_frozen/1e6:.1f}M")
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio},
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        group_by_length=True,
        length_column_name="length",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id or 0),
    )
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    print("[done] saved", final)


if __name__ == "__main__":
    main()
