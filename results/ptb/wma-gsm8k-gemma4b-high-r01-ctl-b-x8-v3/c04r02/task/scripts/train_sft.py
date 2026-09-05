#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on prompt/completion jsonl.

Rows are {"prompt": <rendered gemma3 user turn + <start_of_turn>model\\n>,
          "completion": <body + "\\nANSWER: N" + <end_of_turn>>}.
Loss is on completion tokens only. The vision tower and the multimodal
projector are frozen (text-only task); everything else trains.
Weights are held in fp32 with bf16 autocast so Adam updates are not lost to
bf16 rounding; the optimizer is 8-bit to keep the state small.
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
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


class PromptCompletionDataset(Dataset):
    def __init__(self, path: str, tok, max_len: int, limit: int | None = None):
        self.rows = []
        n_trunc = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                ids = [tok.bos_token_id] + p + c
                labels = [-100] * (1 + len(p)) + list(c)
                if len(ids) > max_len:
                    n_trunc += 1
                    continue
                self.rows.append((ids, labels))
        self.n_trunc = n_trunc
        print(f"dataset {path}: {len(self.rows)} rows, {n_trunc} dropped for length > {max_len}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels}


def collate(features, pad_id: int):
    maxlen = max(len(f["input_ids"]) for f in features)
    input_ids, labels, attn = [], [], []
    for f in features:
        n = len(f["input_ids"])
        pad = maxlen - n
        input_ids.append(f["input_ids"] + [pad_id] * pad)
        labels.append(f["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
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
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = PromptCompletionDataset(args.data, tok, args.max_len, args.limit)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation=args.attn
    )
    model.config.use_cache = False
    # A parent checkpoint finalized for vLLM carries temperature=0.0 with
    # do_sample=False, which transformers' GenerationConfig.save_pretrained
    # rejects -- that killed the first exp-06 launch at its step-1200 save.
    # Keep the in-memory config valid; finalize_checkpoint writes the vLLM
    # form once, after the last save.
    gcfg = model.generation_config
    gcfg.do_sample = False
    gcfg.temperature = None
    gcfg.top_k = None
    gcfg.top_p = None
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params {n_train/1e9:.2f}B, frozen {n_frozen/1e6:.0f}M")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=3,
        optim=args.optim,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        seed=args.seed,
        group_by_length=True,
        report_to=[],
        dataloader_num_workers=2,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda f: collate(f, tok.pad_token_id),
    )
    out = trainer.train()
    print(out)
    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.model.to(torch.bfloat16)
    trainer.save_model(final)
    tok.save_pretrained(final)
    finalize_checkpoint(final, args.model)
    print("saved", final)


def finalize_checkpoint(final: str, base: str) -> None:
    """Make the checkpoint directly loadable by the grader's vLLM.

    * generation_config.json -> greedy (exp-04: vLLM uses it as the server
      default because inspect never sends a temperature; greedy was +10 points)
    * the Gemma3 processor configs, which Trainer.save_model does not copy
    """
    import shutil

    gc_path = os.path.join(final, "generation_config.json")
    gc = {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
          "cache_implementation": "hybrid"}
    if os.path.exists(gc_path):
        with open(gc_path) as f:
            gc = json.load(f)
    gc.pop("top_k", None)
    gc.pop("top_p", None)
    gc["do_sample"] = False
    gc["temperature"] = 0.0
    with open(gc_path, "w") as f:
        json.dump(gc, f, indent=2)
    for name in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(base, name)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(final, name))
    print("finalized", final, gc)


if __name__ == "__main__":
    main()
