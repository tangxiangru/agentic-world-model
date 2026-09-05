#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on GSM8K-style math CoT data."""
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

BASE = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d",
)

START_USER = "<start_of_turn>user\n"
START_MODEL = "<start_of_turn>model\n"
END_TURN = "<end_of_turn>\n"


class SFTData(Dataset):
    def __init__(self, path, tok, max_len, limit=0, fewshot_prob=0.0, seed=0):
        self.tok = tok
        self.max_len = max_len
        rows = [json.loads(l) for l in open(path)]
        rng = random.Random(seed)
        rng.shuffle(rows)
        if limit:
            rows = rows[:limit]
        self.rows = rows
        self.fewshot_prob = fewshot_prob
        self.rng = rng
        self.bos = tok.bos_token  # <bos>
        # pool for random few-shot prefixes (uses only training-split problems)
        self.pool = rows[: min(2000, len(rows))]
        self.lengths = None

    def __len__(self):
        return len(self.rows)

    def token_lengths(self):
        texts = [r["prompt"] + r["completion"] for r in self.rows]
        out = []
        B = 2000
        for i in range(0, len(texts), B):
            enc = self.tok(texts[i:i + B], add_special_tokens=False)["input_ids"]
            out.extend(len(e) + 8 for e in enc)
        return out

    def _build(self, r, shots):
        prefix = ""
        if shots:
            parts = []
            for s in shots:
                parts.append(f"{s['question']}\n\nReasoning:\n{s['completion']}")
            prefix = "\n\n".join(parts) + "\n\n"
        user = prefix + r["prompt"]
        head = self.bos + START_USER + user + "<end_of_turn>\n" + START_MODEL
        body = r["completion"] + "<end_of_turn>"
        h = self.tok(head, add_special_tokens=False)["input_ids"]
        b = self.tok(body, add_special_tokens=False)["input_ids"]
        ids = h + b
        labels = [-100] * len(h) + b[:]
        if len(ids) > self.max_len:
            ids = ids[-self.max_len:]
            labels = labels[-self.max_len:]
        return {"input_ids": ids, "labels": labels}

    def __getitem__(self, i):
        r = self.rows[i]
        shots = []
        if self.fewshot_prob and random.random() < self.fewshot_prob:
            k = random.randint(1, 3)
            shots = [self.pool[random.randrange(len(self.pool))] for _ in range(k)]
        return self._build(r, shots)


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        n = ((n + 63) // 64) * 64
        input_ids, labels, attn = [], [], []
        for f in feats:
            p = n - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * p)
            labels.append(f["labels"] + [-100] * p)
            attn.append([1] * len(f["input_ids"]) + [0] * p)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


class BucketSampler(torch.utils.data.Sampler):
    """Shuffle, then sort within large megabatches to cut padding waste."""

    def __init__(self, lengths, batch_size, mega=64, seed=0):
        self.lengths = lengths
        self.bs = batch_size
        self.mega = mega * batch_size
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        g = random.Random(self.seed + self.epoch)
        self.epoch += 1
        idx = list(range(len(self.lengths)))
        g.shuffle(idx)
        out = []
        for i in range(0, len(idx), self.mega):
            chunk = sorted(idx[i: i + self.mega], key=lambda j: self.lengths[j])
            batches = [chunk[k: k + self.bs] for k in range(0, len(chunk), self.bs)]
            g.shuffle(batches)
            for b in batches:
                out.extend(b)
        return iter(out)


class BucketTrainer(Trainer):
    bucket_sampler = None

    def _get_train_sampler(self, *a, **kw):
        if self.bucket_sampler is not None:
            return self.bucket_sampler
        return super()._get_train_sampler(*a, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_v1.jsonl")
    ap.add_argument("--out", default="runs/sft_v1")
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--fewshot-prob", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--fp32-master", action="store_true")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--no-gc", action="store_true")
    ap.add_argument("--group-by-length", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)

    from liger_kernel.transformers import apply_liger_kernel_to_gemma3

    apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init,
        dtype=torch.float32 if args.fp32_master else torch.bfloat16,
        attn_implementation=args.attn,
    )
    model.config.use_cache = False
    # freeze the vision stack (text-only task)
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"frozen vision params: {n_frozen/1e6:.1f}M")
    print(f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e9:.2f}B")

    ds = SFTData(args.data, tok, args.max_len, args.limit, args.fewshot_prob, args.seed)
    print(f"dataset size: {len(ds)}")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
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
        logging_steps=20,
        save_strategy="no",
        gradient_checkpointing=not args.no_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit" if args.fp32_master else "adamw_torch_fused",
        dataloader_num_workers=8,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = BucketTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    if args.group_by_length:
        import numpy as np
        lens = ds.token_lengths()
        print("median len", float(np.median(lens)))
        trainer.bucket_sampler = BucketSampler(lens, args.bs, seed=args.seed)
    trainer.train()
    print("PEAK MEM GB", torch.cuda.max_memory_allocated()/1e9)

    if args.no_save:
        return
    model.config.use_cache = True
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    # copy processor/preprocessor configs so vLLM can load the multimodal model
    import shutil
    for f in ["preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(BASE, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    print("saved to", args.out)


if __name__ == "__main__":
    main()
