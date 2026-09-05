#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on GSM8K-style math CoT data."""
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

from common import user_prompt, fewshot_system_message

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def build_tokenizer(base=BASE):
    tok = AutoTokenizer.from_pretrained(base)
    with open("templates/gemma3.jinja") as f:
        tok.chat_template = f.read()
    return tok


class SFTData(Dataset):
    def __init__(self, path, tok, max_len=1024, fewshot_prob=0.0, seed=0,
                 fewshot_max_len=2560):
        self.tok = tok
        self.max_len = max_len
        self.fewshot_max_len = fewshot_max_len
        rng = random.Random(seed)
        rows = []
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
        fs_sys = fewshot_system_message()
        self.examples = []
        n_skip = 0
        for r in rows:
            use_fs = rng.random() < fewshot_prob
            msgs = []
            if use_fs:
                msgs.append({"role": "system", "content": fs_sys})
            msgs.append({"role": "user", "content": user_prompt(r["question"])})
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            completion = r["response"].strip() + "<end_of_turn>"
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tok(completion, add_special_tokens=False)["input_ids"]
            limit = self.fewshot_max_len if use_fs else max_len
            if len(p_ids) + len(c_ids) > limit:
                n_skip += 1
                continue
            self.examples.append((p_ids, c_ids))
        print(f"dataset: {len(self.examples)} examples ({n_skip} skipped for length)")
        self.lengths = [len(a) + len(b) for a, b in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        p, c = self.examples[i]
        ids = p + c
        labels = [-100] * len(p) + list(c)
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


def _chunk_ce(hidden, labels, weight):
    logits = torch.nn.functional.linear(hidden, weight).float()
    return torch.nn.functional.cross_entropy(logits, labels, reduction="sum")


class ChunkedCETrainer(Trainer):
    """Computes CE only on supervised positions, in checkpointed chunks.

    Avoids materialising a (B, T, 262144) logits tensor, which otherwise
    dominates memory for Gemma-3's very large vocabulary.
    """

    chunk = 2048

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.model_accepts_loss_kwargs = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        base = self.accelerator.unwrap_model(model)
        hidden = base.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )[0]
        h = hidden[:, :-1, :]
        y = labels[:, 1:]
        mask = y != -100
        h = h[mask]
        y = y[mask]
        w = base.lm_head.weight
        total = None
        for i in range(0, h.shape[0], self.chunk):
            part = torch.utils.checkpoint.checkpoint(
                _chunk_ce, h[i : i + self.chunk], y[i : i + self.chunk], w,
                use_reentrant=False,
            )
            total = part if total is None else total + part
        if total is None:
            total = hidden.sum() * 0.0
        denom = num_items_in_batch if num_items_in_batch is not None else max(y.numel(), 1)
        if torch.is_tensor(denom):
            denom = denom.to(total.device)
        return total / denom


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            n = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * n)
            labels.append(f["labels"] + [-100] * n)
            attn.append([1] * len(f["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_v1.jsonl")
    ap.add_argument("--out", default="runs/sft_v1")
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--fewshot-prob", type=float, default=0.06)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--save-epochs", action="store_true")
    args = ap.parse_args()

    tok = build_tokenizer()
    ds = SFTData(args.data, tok, max_len=args.max_len, fewshot_prob=args.fewshot_prob,
                 seed=args.seed)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False
    # freeze the vision stack: this run is text-only
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector") \
           or name.startswith("vision_tower") or name.startswith("multi_modal_projector"):
            p.requires_grad_(False)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_tr/1e9:.2f}B")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        max_grad_norm=1.0,
        adam_beta2=0.95,
        logging_steps=5,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        save_strategy="steps" if args.save_steps else ("epoch" if args.save_epochs else "no"),
        save_steps=args.save_steps or 500,
        save_total_limit=3,
        report_to=[],
        seed=args.seed,
        optim=args.optim,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = ChunkedCETrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.config.use_cache = True
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    import shutil
    for fn in ("preprocessor_config.json", "processor_config.json", "generation_config.json"):
        src = os.path.join(BASE, fn)
        if os.path.exists(src) and not os.path.exists(os.path.join(args.out, fn)):
            shutil.copy(src, os.path.join(args.out, fn))
    print("saved to", args.out)


if __name__ == "__main__":
    main()
