#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt (text tower) on GSM8K-style math CoT."""
from __future__ import annotations

import argparse
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

BASE = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)


def build_prompt(system: str, user: str) -> str:
    """Mirror templates/gemma3.jinja exactly (single user turn + gen prompt)."""
    prefix = (system.strip() + "\n\n") if system.strip() else ""
    return "<bos><start_of_turn>user\n" + prefix + user.strip() + "<end_of_turn>\n<start_of_turn>model\n"


class SFTData(Dataset):
    def __init__(self, path: str, tok, max_len: int, limit: int | None = None):
        self.rows = []
        skipped = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and len(self.rows) >= limit:
                    break
                r = json.loads(line)
                p_ids = tok(build_prompt(r.get("system", ""), r["user"]), add_special_tokens=False)["input_ids"]
                t_ids = tok(r["assistant"].strip() + "<end_of_turn>", add_special_tokens=False)["input_ids"]
                if len(p_ids) + len(t_ids) > max_len:
                    skipped += 1
                    continue
                self.rows.append((p_ids, t_ids))
        print(f"loaded {len(self.rows)} rows (skipped {skipped} over {max_len} tokens)")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, t = self.rows[i]
        ids = p + t
        labels = [-100] * len(p) + list(t)
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        n = ((n + 63) // 64) * 64  # bucket lengths to reduce recompiles/fragmentation
        input_ids, labels, attn = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
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
    ap.add_argument("--data", default="data/sft_v1.jsonl")
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--out", default="ckpt/sft_v1")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--no-liger", action="store_true")
    ap.add_argument("--attn", default="flash_attention_2")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    ds = SFTData(args.data, tok, args.max_len, args.limit)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    if not args.no_liger:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3

        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("liger kernel applied")

    # Text-only task: freeze the vision stack.
    n_train = 0
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)
        else:
            n_train += p.numel()
    print(f"trainable params: {n_train/1e9:.2f}B")
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        adam_beta2=0.95,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        optim="adamw_torch_fused",
        report_to=[],
        dataloader_num_workers=4,
        remove_unused_columns=False,
        seed=17,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()

    model.config.use_cache = True
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    # keep processor/vision preprocessing files so vLLM loads the model cleanly
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            with open(src) as f_in, open(os.path.join(args.out, fn), "w") as f_out:
                f_out.write(f_in.read())
    print("saved to", args.out)


if __name__ == "__main__":
    main()
