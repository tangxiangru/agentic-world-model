#!/usr/bin/env python3
"""SFT gemma-3-4b-pt on GSM8K-style math CoT data, in the exact eval prompt format."""
from __future__ import annotations

import argparse
import json
import math
import os
import random

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/home/ben/hf_cache/hub")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def build_fewshot_pool(path: str = "data/gsm8k_train.jsonl", limit: int = 3000):
    pool = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            r = json.loads(line)
            pool.append(f"{r['question']}\n\nReasoning:\n{r['solution']}\n\nANSWER: {r['answer']}")
    return pool


class SFTData(Dataset):
    def __init__(self, records, tok, fewshot_pool, fewshot_prob=0.15, max_len=2560, seed=0):
        self.max_len = max_len
        rng = random.Random(seed)
        eot = "<end_of_turn>"
        prompts, targets = [], []
        for r in records:
            user = PROMPT_TEMPLATE.format(prompt=r["question"])
            msgs = []
            if rng.random() < fewshot_prob and fewshot_pool:
                k = rng.randint(1, 10)
                shots = rng.sample(fewshot_pool, k)
                msgs.append({"role": "system", "content": "\n\n".join(shots)})
            msgs.append({"role": "user", "content": user})
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            targets.append(f"{r['solution'].strip()}\n\nANSWER: {r['answer']}{eot}\n")

        p_enc = tok(prompts, add_special_tokens=False)["input_ids"]
        t_enc = tok(targets, add_special_tokens=False)["input_ids"]
        self.examples = []
        for p_ids, t_ids in zip(p_enc, t_enc):
            ids = p_ids + t_ids
            labels = [-100] * len(p_ids) + list(t_ids)
            if len(ids) > max_len:
                ids, labels = ids[-max_len:], labels[-max_len:]
            if all(l == -100 for l in labels):
                continue
            self.examples.append({"input_ids": ids, "labels": labels})
        self.lengths = [len(e["input_ids"]) for e in self.examples]
        print(f"dataset: {len(self.examples)} examples, {sum(self.lengths)/1e6:.1f}M tokens, "
              f"max {max(self.lengths)}, mean {sum(self.lengths)/len(self.lengths):.0f}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        n = (n + 7) // 8 * 8
        input_ids, labels, attn = [], [], []
        for f in feats:
            d = n - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * d)
            labels.append(f["labels"] + [-100] * d)
            attn.append([1] * len(f["input_ids"]) + [0] * d)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


def load_jsonl(path, limit=None):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="jsonl files (optionally file:limit)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--fewshot-prob", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = []
    for spec in args.data:
        if ":" in spec:
            path, lim = spec.rsplit(":", 1)
            lim = int(lim)
        else:
            path, lim = spec, None
        rs = load_jsonl(path, lim)
        print(f"{path}: {len(rs)}")
        records += rs
    random.Random(args.seed).shuffle(records)
    print("total records", len(records))

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = open("templates/gemma3.jinja").read()

    ds = SFTData(records, tok, build_fewshot_pool(), args.fewshot_prob, args.max_len, args.seed)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init, dtype=torch.float32, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    for n, p in model.named_parameters():
        if "vision_tower" in n or "multi_modal_projector" in n:
            p.requires_grad_(False)

    targs = TrainingArguments(
        output_dir=args.out + "_tmp",
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        use_liger_kernel=True,
        group_by_length=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=Collator(tok.pad_token_id))
    trainer.train()

    model.config.use_cache = True
    model = model.to(torch.bfloat16)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    # copy processor/config extras so vllm loads it like the base model
    import shutil
    for fn in ["preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, fn))
    print("saved to", args.out)


if __name__ == "__main__":
    main()
