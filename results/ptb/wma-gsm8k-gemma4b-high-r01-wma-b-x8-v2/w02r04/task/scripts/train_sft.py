#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on pre-rendered prompt/completion jsonl.

The jsonl rows already contain the exact strings the grader's chat template
produces (see scripts/common_fmt.py), so nothing is re-templated here: the
prompt is masked out of the loss and the completion (which ends with
<end_of_turn>) is the target.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class JsonlSFT(Dataset):
    def __init__(self, path, tok, max_seq_len, limit=None):
        self.rows = []
        n_trunc = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                if len(p) + len(c) > max_seq_len:
                    n_trunc += 1
                    continue
                self.rows.append((p, c))
        self.n_trunc = n_trunc
        self.lengths = [len(p) + len(c) for p, c in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        ids = p + c
        labels = [-100] * len(p) + list(c)
        return {"input_ids": ids, "labels": labels}


class TokenBudgetBatches:
    """Length-sorted batches capped by padded token count.

    p50 row length is ~310 tokens and p95 ~2360 (the 10-shot rows), so a fixed
    per-device batch size either wastes the GPU on short rows or OOMs on long
    ones. Batches are built so that len(batch) * max_len(batch) <= budget.
    """

    def __init__(self, lengths, budget, seed=0, chunk=2048):
        self.batches = []
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        cur, cur_max = [], 0
        for i in order:
            m = max(cur_max, lengths[i])
            if cur and m * (len(cur) + 1) > budget:
                self.batches.append(cur)
                cur, cur_max = [i], lengths[i]
            else:
                cur.append(i)
                cur_max = m
        if cur:
            self.batches.append(cur)
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        g = random.Random(self.seed + self.epoch)
        b = list(self.batches)
        g.shuffle(b)
        self.epoch += 1
        return iter(b)


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, labels, mask = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            mask.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2304)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-strategy", default="epoch")
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--token-budget", type=int, default=8192)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = JsonlSFT(args.data, tok, args.max_seq_len, args.limit)
    print(f"rows kept {len(ds)}  dropped(too long) {ds.n_trunc}  "
          f"p50 {sorted(ds.lengths)[len(ds)//2]}  max {max(ds.lengths)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False
    # A parent finalized with --decode greedy carries do_sample=False + temperature=0.0,
    # which GenerationConfig.save_pretrained rejects: it killed exp-05's first launch at
    # the step-400 save. Training never reads these; finalize_ckpt.py rewrites them after.
    gc = model.generation_config
    if getattr(gc, "temperature", None) is not None and not getattr(gc, "do_sample", True):
        gc.do_sample = True
        gc.temperature = None
        print("sanitized generation_config for saving", flush=True)
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    # no images in this corpus: freeze the vision tower / projector outright
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"frozen vision params: {n_frozen/1e6:.1f}M", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=4,
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        optim="adamw_torch_fused",
        seed=args.seed,
        save_safetensors=True,
        accelerator_config={"split_batches": False, "dispatch_batches": False},
    )

    batcher = TokenBudgetBatches(ds.lengths, args.token_budget, seed=args.seed)
    sizes = [len(b) for b in batcher.batches]
    print(f"micro-batches/epoch {len(batcher)}  batch size min/median/max "
          f"{min(sizes)}/{sorted(sizes)[len(sizes)//2]}/{max(sizes)}", flush=True)
    collator = Collator(tok.pad_token_id or 0)

    class T(Trainer):
        def get_train_dataloader(self):
            from torch.utils.data import DataLoader
            return self.accelerator.prepare(DataLoader(
                ds, batch_sampler=batcher, collate_fn=collator,
                num_workers=2, pin_memory=True,
            ))

    trainer = T(model=model, args=targs, train_dataset=ds,
                data_collator=collator)
    trainer.train()
    trainer.save_model(os.path.join(args.out, "final"))
    tok.save_pretrained(os.path.join(args.out, "final"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
