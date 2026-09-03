#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on a pre-rendered {prompt, completion} jsonl.

The jsonl rows are already rendered with the grader's chat template, so this
script only tokenises and masks: loss is taken on the completion tokens alone.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"


class JsonlSFT(Dataset):
    def __init__(self, path, tok, max_len):
        self.rows = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                ids = (p + c)[:max_len]
                labels = ([-100] * len(p) + c)[:max_len]
                if all(x == -100 for x in labels):
                    continue
                self.rows.append({"input_ids": ids, "labels": labels})
        self.lengths = [len(r["input_ids"]) for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, lab, att = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            lab.append(f["labels"] + [-100] * k)
            att.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(lab, dtype=torch.long),
            "attention_mask": torch.tensor(att, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=SNAPSHOT)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1408)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    with open(TEMPLATE_PATH) as f:
        tok.chat_template = f.read()

    ds = JsonlSFT(args.data, tok, args.max_len)
    print(f"train rows: {len(ds)}  max_len={max(ds.lengths)}", flush=True)

    cfg = AutoConfig.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="eager", config=cfg
        )
    except Exception as e:  # pragma: no cover
        print("eager load failed, retrying default attn:", e, flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)

    # freeze everything that is not the text decoder
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {n_frozen/1e6:.0f}M, trainable {trainable/1e6:.0f}M", flush=True)

    model.config.use_cache = False

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
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        report_to=[],
        seed=args.seed,
        optim=args.optim,
        dataloader_num_workers=4,
        save_safetensors=True,
    )

    # group_by_length needs a length signal; provide it explicitly
    class DS(Dataset):
        def __init__(self, inner):
            self.inner = inner

        def __len__(self):
            return len(self.inner)

        def __getitem__(self, i):
            r = dict(self.inner[i])
            return r

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    out = trainer.train()
    print(out, flush=True)

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    for fn in ("preprocessor_config.json", "processor_config.json", "added_tokens.json",
               "tokenizer.model", "generation_config.json"):
        src = os.path.join(args.model, fn)
        if os.path.exists(src) and not os.path.exists(os.path.join(final, fn)):
            shutil.copy(src, os.path.join(final, fn))
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
