#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt for GSM8K.

Rows are rendered with scripts/prompting.py, which check_render.py proves is
byte-identical to templates/gemma3.jinja (the template the grader uses).
Prompt tokens are masked to -100; every target ends on <end_of_turn> (id 106),
which is in the checkpoint's generation_config eos list, so vLLM stops there.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from prompting import fewshot_block, render_prompt, render_target  # noqa: E402

SNAP = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
GSM8K_TRAIN = ("/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/"
               "740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet")


def load_fewshot_pool():
    import pyarrow.parquet as pq
    rows = pq.read_table(GSM8K_TRAIN).to_pylist()
    pool = []
    for r in rows[:2000]:
        ans = r["answer"]
        reasoning, _, target = ans.rpartition("####")
        pool.append((r["question"].strip(), reasoning.strip(), target.strip()))
    return pool


class SFTData(Dataset):
    def __init__(self, path, tok, max_len, fewshot_p, fewshot_ks, seed, limit=None):
        self.rows = []
        rng = random.Random(seed)
        pool = load_fewshot_pool() if fewshot_p > 0 else []
        n_trunc = 0
        with open(path) as f:
            raw = [json.loads(l) for l in f]
        if limit:
            raw = raw[:limit]
        for r in raw:
            system = None
            if pool and rng.random() < fewshot_p:
                k = rng.choice(fewshot_ks)
                shots = rng.sample(pool, k)
                system = "\n\n".join(fewshot_block(*s) for s in shots)
            p = render_prompt(r["question"], system)
            t = r["target"] if "target" in r else render_target(r["answer"])
            pids = tok(p, add_special_tokens=False)["input_ids"]
            tids = tok(t, add_special_tokens=False)["input_ids"]
            if len(pids) + len(tids) > max_len:
                n_trunc += 1
                continue
            self.rows.append((pids, tids))
        self.n_trunc = n_trunc
        self.lengths = [len(a) + len(b) for a, b in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        pids, tids = self.rows[i]
        return {"input_ids": pids + tids,
                "labels": [-100] * len(pids) + tids}


def token_budget_batches(lengths, budget, max_bs, seed):
    """Group similar-length rows so that padded tokens per micro-batch stay under
    `budget`.  Gemma's 262k vocab makes the logits tensor the memory bottleneck,
    so a fixed row count either OOMs on long rows or wastes the GPU on short ones."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lengths[i])
        if cur and (m * (len(cur) + 1) > budget or len(cur) >= max_bs):
            batches.append(cur)
            cur, cur_max = [i], lengths[i]
        else:
            cur, cur_max = cur + [i], m
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    return batches


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    ids, labels, mask = [], [], []
    for b in batch:
        n = m - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        mask.append([1] * len(b["input_ids"]) + [0] * n)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_main.jsonl")
    ap.add_argument("--parent", default=SNAP)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=1792)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--token-budget", type=int, default=6144)
    ap.add_argument("--max-bs", type=int, default=48)
    ap.add_argument("--fewshot-p", type=float, default=0.2)
    ap.add_argument("--fewshot-ks", default="2,4,8")
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--optim", default="adamw_torch_fused")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAP)
    ds = SFTData(args.data, tok, args.max_len, args.fewshot_p,
                 [int(x) for x in args.fewshot_ks.split(",")], args.seed, args.limit)
    ls = sorted(ds.lengths)
    print(f"rows={len(ds)} dropped_over_max_len={ds.n_trunc} "
          f"p50={ls[len(ls)//2]} p99={ls[int(len(ls)*0.99)]} max={ls[-1]} "
          f"total_tokens={sum(ls)/1e6:.1f}M", flush=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent,
        dtype=(torch.bfloat16 if args.dtype == "bf16" else torch.float32),
        attn_implementation=args.attn)
    for p in model.model.vision_tower.parameters():
        p.requires_grad_(False)
    for p in model.model.multi_modal_projector.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e9:.2f}B", flush=True)

    batches = token_budget_batches(ds.lengths, args.token_budget, args.max_bs, args.seed)
    print(f"micro-batches/epoch={len(batches)} mean_rows/batch={len(ds)/len(batches):.1f} "
          f"optimizer_steps/epoch={len(batches)//args.accum}", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy=("steps" if args.save_steps else "no"),
        save_steps=(args.save_steps or 500),
        save_total_limit=2,
        gradient_checkpointing=bool(args.grad_ckpt),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=False,
        optim=args.optim,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    class BudgetTrainer(Trainer):
        def get_train_dataloader(self):
            from torch.utils.data import DataLoader
            return DataLoader(
                self.train_dataset,
                batch_sampler=batches,
                collate_fn=self.data_collator,
                num_workers=2,
                pin_memory=True,
            )

    trainer = BudgetTrainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    # the grader loads final_model/ with gpu_memory_utilization=0.3 (24 GB); ship
    # bf16 weights like the starting revision, not the fp32 master copy
    model.to(torch.bfloat16)
    model.config.torch_dtype = "bfloat16"
    trainer.save_model(final)
    tok.save_pretrained(final)
    try:
        AutoProcessor.from_pretrained(SNAP).save_pretrained(final)
    except Exception as e:  # processor is optional for a text-only grader
        print("processor save failed:", e)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
