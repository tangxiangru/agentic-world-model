#!/usr/bin/env python3
"""Completion-only SFT of gemma-3-4b-pt for BFCL tool calling.

Prompt/completion are pre-rendered with the grader's chat template. We tokenize
with add_special_tokens=False (prompt already carries <bos>) and mask prompt
tokens so loss is only on the completion (ending in <end_of_turn>).
"""
import argparse
import json

import sys

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class StderrLossLogger(TrainerCallback):
    """Print loss to stderr (unbuffered) so it is visible in redirected logs."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"[step {state.global_step}] loss={logs['loss']:.4f} "
                  f"lr={logs.get('learning_rate', 0):.2e} epoch={logs.get('epoch', 0):.2f}",
                  file=sys.stderr, flush=True)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--out", default="work/sft1")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=400)
    args = ap.parse_args()

    tk = AutoTokenizer.from_pretrained(SNAP)
    EOT_ID = tk.convert_tokens_to_ids("<end_of_turn>")

    def tokenize(ex):
        p = tk(ex["prompt"], add_special_tokens=False)["input_ids"]
        c = tk(ex["completion"], add_special_tokens=False)["input_ids"]
        ids = p + c
        labels = [-100] * len(p) + list(c)
        ids = ids[: args.max_seq_len]
        labels = labels[: args.max_seq_len]
        return {"input_ids": ids, "labels": labels}

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(tokenize, remove_columns=ds.column_names, num_proc=8)

    # sanity: every row must have >=1 supervised token and end with <end_of_turn>
    def _ok(ex):
        return any(l != -100 for l in ex["labels"]) and ex["input_ids"][-1] == EOT_ID
    n_before = len(ds)
    ds = ds.filter(_ok, num_proc=8)
    print(f"dataset: {n_before} -> {len(ds)} rows after sanity filter; EOT_ID={EOT_ID}")

    pad_id = tk.pad_token_id

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids = b["input_ids"]; lab = b["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    model = AutoModelForCausalLM.from_pretrained(
        SNAP, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        bf16=True,
        logging_steps=20,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=5,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate,
                      callbacks=[StderrLossLogger()])
    trainer.train()

    # final save with tokenizer + deterministic greedy generation_config that
    # stops on <end_of_turn> (106) — matches the grader.
    final = f"{args.out}/final"
    trainer.save_model(final)
    tk.save_pretrained(final)
    from transformers import GenerationConfig
    gc = GenerationConfig(
        bos_token_id=tk.bos_token_id,
        eos_token_id=[tk.eos_token_id, EOT_ID],
        pad_token_id=tk.pad_token_id,
        do_sample=False,
        cache_implementation="hybrid",
    )
    gc.save_pretrained(final)
    print("saved final ->", final)


if __name__ == "__main__":
    main()
