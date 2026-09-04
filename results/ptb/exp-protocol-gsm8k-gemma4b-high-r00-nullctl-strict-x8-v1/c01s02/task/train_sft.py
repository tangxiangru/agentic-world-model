#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on GSM8K-style math data."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

SNAPSHOT = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
)


def build_tokenizer(template_path="templates/gemma3.jinja"):
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    with open(template_path) as f:
        tok.chat_template = f.read()
    return tok


class SFTData(Dataset):
    def __init__(self, path, tok, max_len=1024, limit=None, fewshot_prob=0.0,
                 fewshot_pool=None, seed=0):
        self.rows = []
        rng = random.Random(seed)
        n_skip = 0
        prompts, completions = [], []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and len(prompts) >= limit:
                    break
                d = json.loads(line)
                prompt = d["prompt"]
                if fewshot_pool and rng.random() < fewshot_prob:
                    k = rng.randint(1, 6)
                    shots = rng.sample(fewshot_pool, k)
                    prompt = "\n\n".join(shots) + "\n\n" + prompt
                prompts.append(
                    "<bos><start_of_turn>user\n" + prompt.strip() +
                    "<end_of_turn>\n<start_of_turn>model\n"
                )
                completions.append(d["completion"].strip())

        eot = tok.convert_tokens_to_ids("<end_of_turn>")
        B = 2000
        for s in range(0, len(prompts), B):
            p_enc = tok(prompts[s:s + B], add_special_tokens=False)["input_ids"]
            c_enc = tok(completions[s:s + B], add_special_tokens=False)["input_ids"]
            for p_ids, c_ids in zip(p_enc, c_enc):
                c_ids = c_ids + [eot]
                ids = p_ids + c_ids
                if len(ids) > max_len:
                    n_skip += 1
                    continue
                self.rows.append((ids, [-100] * len(p_ids) + c_ids))
        print(f"dataset: {len(self.rows)} examples, skipped {n_skip} (>{max_len} tok)")
        self.lengths = [len(r[0]) for r in self.rows]
        print(f"  mean len {sum(self.lengths)/len(self.lengths):.1f}, "
              f"total tokens {sum(self.lengths)/1e6:.1f}M")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


def collate(features, pad_id=0):
    maxlen = max(len(f["input_ids"]) for f in features)
    input_ids, labels, attn = [], [], []
    for f in features:
        n = maxlen - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_id] * n)
        labels.append(f["labels"] + [-100] * n)
        attn.append([1] * len(f["input_ids"]) + [0] * n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_gsm.jsonl")
    ap.add_argument("--out", default="runs/sft1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--fewshot-prob", type=float, default=0.0)
    ap.add_argument("--init", default=SNAPSHOT)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--fp32-master", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--liger", type=int, default=1)
    args = ap.parse_args()

    tok = build_tokenizer()

    fewshot_pool = None
    if args.fewshot_prob > 0:
        from datasets import load_dataset
        tr = load_dataset("openai/gsm8k", "main", split="train")
        fewshot_pool = []
        for r in tr.select(range(2000)):
            q = r["question"]
            a = r["answer"].split("####")
            target = a.pop().strip()
            reasoning = "####".join(a).strip()
            fewshot_pool.append(f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}")

    ds = SFTData(args.data, tok, max_len=args.max_len, limit=args.limit,
                 fewshot_prob=args.fewshot_prob, fewshot_pool=fewshot_pool)

    dtype = torch.float32 if args.fp32_master else torch.bfloat16
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init, dtype=dtype, attn_implementation=args.attn
    )
    if args.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3
        apply_liger_kernel_to_gemma3(
            model=model, fused_linear_cross_entropy=True, cross_entropy=False
        )
        print("applied liger kernels")
    model.config.use_cache = False
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_train/1e9:.2f}B, frozen {n_frozen/1e6:.0f}M")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        logging_steps=5,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps if args.save_steps else 500,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        adam_beta2=0.95,
        weight_decay=0.0,
        max_grad_norm=1.0,
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to=[],
        seed=0,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda f: collate(f, pad_id=tok.pad_token_id or 0),
    )
    t0 = time.time()
    trainer.train()
    print(f"train time {(time.time()-t0)/60:.1f} min")

    model.config.use_cache = True
    model.to(torch.bfloat16)
    model.config.torch_dtype = "bfloat16"
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
