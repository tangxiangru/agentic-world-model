#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on GSM8K-style CoT data.

Trains only the language model (vision tower + multimodal projector are frozen)
but keeps the original Gemma3ForConditionalGeneration architecture so the result
loads in vLLM exactly like the base checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil

import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def make_fewshot_pool(n=400):
    """Few-shot exemplars drawn from the GSM8K *train* split (same source the harness uses)."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    pool = []
    for row in ds.select(range(n)):
        reasoning, _, final = row["answer"].partition("####")
        reasoning = re.sub(r"<<[^>]*>>", "", reasoning).strip()
        pool.append(f"{row['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {final.strip()}")
    return pool


def build_dataset(path, tok, max_len, fewshot_prob, seed, nproc=16):
    from datasets import Dataset

    rows = [json.loads(l) for l in open(path)]
    pool = make_fewshot_pool() if fewshot_prob > 0 else []

    prompts, fulls = [], []
    rng = random.Random(seed)
    for r in rows:
        user = r["messages"][0]["content"].strip()
        assistant = r["messages"][1]["content"].strip()
        if pool and rng.random() < fewshot_prob:
            k = rng.choice([2, 3, 4, 5])
            user = "\n\n".join(rng.sample(pool, k)) + "\n\n" + user
        p = f"<bos><start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
        prompts.append(p)
        fulls.append(f"{p}{assistant}<end_of_turn>\n")

    ds = Dataset.from_dict({"prompt": prompts, "full": fulls})

    def tok_fn(batch):
        p = tok(batch["prompt"], add_special_tokens=False)["input_ids"]
        f = tok(batch["full"], add_special_tokens=False)["input_ids"]
        out_ids, out_lab, out_len = [], [], []
        for pi, fi in zip(p, f):
            fi = fi[:max_len]
            lab = list(fi)
            for j in range(min(len(pi), len(lab))):
                lab[j] = -100
            out_ids.append(fi)
            out_lab.append(lab)
            out_len.append(len(fi))
        return {"input_ids": out_ids, "labels": out_lab, "length": out_len}

    ds = ds.map(
        tok_fn, batched=True, batch_size=1000, num_proc=nproc, remove_columns=["prompt", "full"]
    )
    # drop examples where the completion got truncated away entirely
    ds = ds.filter(lambda b: [any(x != -100 for x in lab) for lab in b["labels"]], batched=True, num_proc=nproc)
    return ds


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            ids = f["input_ids"]
            lab = f["labels"]
            k = n - len(ids)
            input_ids.append(list(ids) + [self.pad_id] * k)
            labels.append(list(lab) + [-100] * k)
            attn.append([1] * len(ids) + [0] * k)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="work/sft_v1.jsonl")
    ap.add_argument("--out", default="work/sft_v1")
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--fewshot-prob", type=float, default=0.15)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--optim", default="adamw_torch_fused")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    ds = build_dataset(args.data, tok, args.max_len, args.fewshot_prob, args.seed)
    print(f"dataset: {len(ds)} examples, tokens={sum(ds['length'])/1e6:.1f}M", flush=True)

    # Fused linear+cross-entropy: gemma3's 262k vocab otherwise materialises
    # a (batch*seqlen, 262144) fp32 logit tensor and OOMs.
    from liger_kernel.transformers import apply_liger_kernel_to_gemma3

    apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)

    config = AutoConfig.from_pretrained(args.init)
    model = AutoModelForImageTextToText.from_pretrained(
        args.init,
        config=config,
        dtype=torch.float32 if args.dtype == "fp32" else torch.bfloat16,
        attn_implementation=args.attn,
    )
    model.config.use_cache = False

    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {n_frozen/1e6:.0f}M, trainable {trainable/1e6:.0f}M", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio},
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 999999,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to=[],
        seed=args.seed,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds.remove_columns([]),
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()

    print("saving ...", flush=True)
    model.config.use_cache = True
    # always ship bf16 weights so vLLM loads at the same footprint as the base model
    model.to(torch.bfloat16)
    model.config.torch_dtype = "bfloat16"
    if hasattr(model.config, "text_config"):
        model.config.text_config.torch_dtype = "bfloat16"
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    try:
        AutoProcessor.from_pretrained(BASE).save_pretrained(args.out)
    except Exception as e:
        print("processor save failed:", e)
    for fn in ["generation_config.json", "preprocessor_config.json", "processor_config.json"]:
        src, dst = os.path.join(BASE, fn), os.path.join(args.out, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    print("done", flush=True)


if __name__ == "__main__":
    main()
