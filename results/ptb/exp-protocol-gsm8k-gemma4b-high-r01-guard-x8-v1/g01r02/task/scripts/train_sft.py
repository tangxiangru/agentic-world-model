#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on a prompt/completion jsonl.

The prompt strings in the jsonl are byte-identical to what templates/gemma3.jinja
renders (verified by scripts/check_render.py), so tokenisation is
add_special_tokens=False throughout and the <bos> comes from the prompt text.
Loss is on completion tokens only.

Two things drive the memory layout:
  * params are fp32 with bf16 autocast. bf16 master weights would silently drop
    an lr=1e-5 Adam step: ULP at |w|~0.02 is 7.8e-5, the step is 1e-5.
  * gemma-3's vocab is 262144, so the logits tensor, not the activations, is the
    peak. Batches are therefore built to a *token* budget, not a fixed size.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil

import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoTokenizer, Gemma3ForConditionalGeneration,
                          Trainer, TrainingArguments)


class Rows(Dataset):
    def __init__(self, path, tok, max_len, limit=None):
        prompts, comps = [], []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                d = json.loads(line)
                prompts.append(d["prompt"])
                comps.append(d["completion"])
        p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
        c_ids = tok(comps, add_special_tokens=False)["input_ids"]
        self.rows, n_trunc = [], 0
        for p, c in zip(p_ids, c_ids):
            if len(p) + len(c) > max_len:
                n_trunc += 1
                continue
            self.rows.append((np.array(p + c, dtype=np.int32), len(p)))
        lens = np.array([len(r[0]) for r in self.rows])
        self.lengths = lens
        print(f"[data] kept {len(self.rows)}, dropped {n_trunc} over {max_len} "
              f"({n_trunc / max(1, len(p_ids)):.3%}); len p50={np.percentile(lens, 50):.0f} "
              f"p99={np.percentile(lens, 99):.0f} max={lens.max()}; "
              f"tokens={lens.sum() / 1e6:.1f}M", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return i


class TokenBudgetBatches:
    """Length-bucketed batches whose padded token count stays under `budget`."""

    def __init__(self, lengths, budget, seed=0, mega=8192, max_bs=64):
        idx = list(range(len(lengths)))
        random.Random(seed).shuffle(idx)
        batches = []
        for s in range(0, len(idx), mega):
            chunk = sorted(idx[s:s + mega], key=lambda i: lengths[i])
            cur, curmax = [], 0
            for i in chunk:
                m = max(curmax, lengths[i])
                if cur and (m * (len(cur) + 1) > budget or len(cur) >= max_bs):
                    batches.append(cur)
                    cur, curmax = [i], lengths[i]
                else:
                    cur, curmax = cur + [i], m
            if cur:
                batches.append(cur)
        random.Random(seed + 1).shuffle(batches)
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class Collator:
    def __init__(self, ds, pad_id):
        self.ds, self.pad_id = ds, pad_id

    def __call__(self, indices):
        rows = [self.ds.rows[i] for i in indices]
        n = max(len(r[0]) for r in rows)
        ids = torch.full((len(rows), n), self.pad_id, dtype=torch.long)
        lab = torch.full((len(rows), n), -100, dtype=torch.long)
        att = torch.zeros((len(rows), n), dtype=torch.long)
        for i, (seq, np_) in enumerate(rows):
            L = len(seq)
            t = torch.from_numpy(seq.astype(np.int64))
            ids[i, :L] = t
            lab[i, np_:L] = t[np_:]
            att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


class BudgetTrainer(Trainer):
    batch_sampler = None

    def get_train_dataloader(self):
        return self.accelerator.prepare(DataLoader(
            self.train_dataset, batch_sampler=self.batch_sampler,
            collate_fn=self.data_collator, num_workers=2, pin_memory=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--budget", type=int, default=6144)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = Rows(args.data, tok, args.max_len, args.limit)
    bs = TokenBudgetBatches(ds.lengths, args.budget, seed=args.seed)
    print(f"[data] {len(bs)} micro-batches, mean size "
          f"{len(ds) / len(bs):.1f}, optim steps ~{len(bs) // args.accum}", flush=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation=args.attn)
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {trainable / 1e9:.2f}B frozen {n_frozen / 1e9:.2f}B", flush=True)
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,  # unused: batch_sampler drives the loader
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        disable_tqdm=True,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 10 ** 9,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        accelerator_config={"dispatch_batches": False, "split_batches": False},
        remove_unused_columns=False,
    )

    trainer = BudgetTrainer(model=model, args=targs, train_dataset=ds,
                            data_collator=Collator(ds, tok.pad_token_id))
    trainer.batch_sampler = bs
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.model.to(torch.bfloat16)
    # HF validates generation_config on save and rejects do_sample=False with
    # temperature=0.0 -- exactly the greedy config exp-03 adopted. Neutralise it
    # for the save, then put the parent's file back verbatim; vLLM reads the json.
    trainer.model.generation_config.do_sample = True
    trainer.model.generation_config.temperature = None
    trainer.model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    for f in ("generation_config.json", "preprocessor_config.json",
              "processor_config.json"):
        src = os.path.join(args.model, f)
        if os.path.exists(src):
            dst = os.path.join(final, f)
            if os.path.exists(dst):
                os.remove(dst)
            shutil.copy(src, dst)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
