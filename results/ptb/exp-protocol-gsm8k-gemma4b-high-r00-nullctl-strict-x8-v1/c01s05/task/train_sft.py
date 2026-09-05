#!/usr/bin/env python3
"""Supervised fine-tuning of gemma-3-4b-pt for GSM8K in the inspect-eval format."""
import argparse
import json
import math
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

from common import SNAPSHOT, get_tokenizer, user_message


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def build_fewshot_pool():
    from datasets import load_dataset
    import re
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    import json as _json, os as _os
    devq = set()
    if _os.path.exists("data/dev.jsonl"):
        devq = {_json.loads(l)["question"] for l in open("data/dev.jsonl")}
    pool = []
    for rec in gsm:
        if rec["question"].strip() in devq:
            continue
        body, tgt = rec["answer"].split("####")
        pool.append(f"{rec['question'].strip()}\n\nReasoning:\n{body.strip()}\n\nANSWER: {tgt.strip()}")
    return pool


class SFTDataset(Dataset):
    def __init__(self, items, tok, fewshot_pool, fewshot_prob, max_len, seed=0):
        self.items = items
        self.tok = tok
        self.pool = fewshot_pool
        self.fewshot_prob = fewshot_prob
        self.max_len = max_len
        self.seed = seed
        self.eot = tok.convert_tokens_to_ids("<end_of_turn>")
        # deterministic per-index rng decisions
        rng = random.Random(seed)
        self.fs = []
        for _ in items:
            if rng.random() < fewshot_prob:
                k = rng.choice([2, 3, 4, 5, 6, 8, 10])
                idxs = rng.sample(range(len(fewshot_pool)), k)
                self.fs.append(idxs)
            else:
                self.fs.append(None)
        prompts, comps = [], []
        for i in range(len(items)):
            p, c = self.render(i)
            prompts.append(p)
            comps.append(c)
        print("tokenizing...", flush=True)
        p_enc = tok(prompts, add_special_tokens=False)["input_ids"]
        c_enc = tok(comps, add_special_tokens=False)["input_ids"]
        self.examples = []
        for p_ids, c_ids in zip(p_enc, c_enc):
            c_ids = c_ids + [self.eot]
            ids = p_ids + c_ids
            n_p = len(p_ids)
            if len(ids) > max_len:
                cut = len(ids) - max_len
                ids = ids[cut:]
                n_p = max(0, n_p - cut)
            self.examples.append((ids, n_p))
        self.lengths = [len(e[0]) for e in self.examples]
        print(f"dataset: {len(self.examples)} ex, {sum(self.lengths)/1e6:.1f}M tokens, "
              f"max {max(self.lengths)}, mean {sum(self.lengths)/len(self.lengths):.0f}", flush=True)

    def render(self, i):
        it = self.items[i]
        msgs = []
        if self.fs[i] is not None:
            sysmsg = "\n\n".join(self.pool[j] for j in self.fs[i])
            msgs.append({"role": "system", "content": sysmsg})
        msgs.append({"role": "user", "content": user_message(it["question"])})
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        completion = f"{it['solution'].strip()}\n\nANSWER: {it['answer']}"
        return prompt, completion

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, n_p = self.examples[i]
        labels = [-100] * n_p + list(ids[n_p:])
        return {"input_ids": list(ids), "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        n = min(int(math.ceil(n / 64) * 64), 4096)
        input_ids, labels, attn = [], [], []
        for f in feats:
            ids = f["input_ids"][:n]
            lb = f["labels"][:n]
            pad = n - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lb + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--init", default=SNAPSHOT)
    ap.add_argument("--out", default="ckpt/sft1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--fewshot-prob", type=float, default=0.3)
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--liger", type=int, default=1)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = get_tokenizer()
    items = load_jsonl(args.data)
    if args.max_samples > 0:
        items = items[: args.max_samples]
    pool = build_fewshot_pool() if args.fewshot_prob > 0 else []
    ds = SFTDataset(items, tok, pool, args.fewshot_prob, args.max_len, seed=args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.init,
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    model.config.use_cache = False
    # freeze vision tower / projector (unused for text-only GSM8K)
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e9:.2f}B")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio},
        warmup_steps=args.warmup,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        adam_beta2=0.95,
        weight_decay=0.0,
        max_grad_norm=1.0,
        report_to=[],
        dataloader_num_workers=4,
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        seed=args.seed,
        use_liger_kernel=args.liger,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id or 0),
    )
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.config.use_cache = True
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    # copy processor/vision configs so the checkpoint loads exactly like the base
    import shutil
    for fn in ["preprocessor_config.json", "processor_config.json", "generation_config.json"]:
        src = os.path.join(SNAPSHOT, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, fn))
    print("saved to", args.out)


if __name__ == "__main__":
    main()
