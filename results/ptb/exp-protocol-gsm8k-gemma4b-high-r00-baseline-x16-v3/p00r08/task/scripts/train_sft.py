#!/usr/bin/env python3
"""Completion-only SFT of gemma-3-4b-pt on GSM8K-style data.

Prompts are rendered with the grader's own chat template and prompt wrapper
(see common.py). Loss is computed on the target only; every target ends with
<end_of_turn>, the token vLLM stops on (generation_config eos_token_id [1,106]).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class SFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_seq_len, fewshot_p, seed, max_shots=10):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []
        rng = random.Random(seed)
        pool = common.eval_fewshot_samples()
        blocks = [common.fewshot_block(s) for s in pool]
        questions = [s.input.strip() for s in pool]

        n_trunc = 0
        for r in rows:
            q = r["prompt_question"]
            system = None
            if rng.random() < fewshot_p:
                k = rng.randint(1, max_shots)
                idx = list(range(len(blocks)))
                rng.shuffle(idx)
                # never show the row's own question as a solved example
                idx = [i for i in idx if questions[i] != q][:k]
                if rng.random() < 0.5:
                    # the grader's exact prefix: all ten, in the grader's order
                    idx = [i for i in range(len(blocks)) if questions[i] != q]
                system = "\n\n".join(blocks[i] for i in idx)

            prompt = common.render_prompt(self.tok, q, system)
            prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
            target_ids = self.tok(r["target"], add_special_tokens=False)["input_ids"]
            if target_ids[-1] != common.STOP_TOKEN_ID:
                raise ValueError(
                    f"target does not end in {common.STOP_TOKEN!r} "
                    f"(id {common.STOP_TOKEN_ID}): {r['target'][-60:]!r}"
                )

            ids = prompt_ids + target_ids
            if len(ids) > max_seq_len:
                n_trunc += 1
                continue
            labels = [-100] * len(prompt_ids) + list(target_ids)
            self.examples.append({"input_ids": ids, "labels": labels})
        self.n_trunc = n_trunc

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

    def lengths(self):
        return [len(e["input_ids"]) for e in self.examples]


class BatchedDataset(Dataset):
    """Groups examples into token-budgeted batches.

    Gemma-3's vocabulary is 262144, so the fp32 logits tensor is
    batch*seq*262144*4 bytes -- 46 GiB for 16 rows of 3k tokens, which is what
    OOMed the first smoke run. A fixed row count cannot bound that; a token
    budget can, and it also keeps short batches wide enough to feed the GPU.
    """

    def __init__(self, base: "SFTDataset", token_budget: int, seed: int):
        order = sorted(range(len(base)), key=lambda i: len(base[i]["input_ids"]))
        self.batches = []
        cur = []
        cur_max = 0
        for i in order:
            n = len(base[i]["input_ids"])
            new_max = max(cur_max, n)
            if cur and new_max * (len(cur) + 1) > token_budget:
                self.batches.append(cur)
                cur, cur_max = [i], n
            else:
                cur.append(i)
                cur_max = new_max
        if cur:
            self.batches.append(cur)
        random.Random(seed).shuffle(self.batches)
        self.base = base

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return [self.base[j] for j in self.batches[i]]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        if len(batch) == 1 and isinstance(batch[0], list):
            batch = batch[0]
        n = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def copy_aux_files(src: Path, dst: Path):
    """Everything vLLM needs besides the weights (tokenizer, processor, ...)."""
    for name in [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "preprocessor_config.json",
        "processor_config.json",
    ]:
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--fewshot-p", type=float, default=0.25)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--master-fp32", type=int, default=1)
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--token-budget", type=int, default=8192)
    args = ap.parse_args()

    from transformers import (
        AutoTokenizer,
        Gemma3ForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"chat template sha256[:16] = {common.template_hash()}", flush=True)

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    ds = SFTDataset(rows, tok, args.max_seq_len, args.fewshot_p, args.seed)
    L = sorted(ds.lengths())
    print(
        f"examples {len(ds)} (dropped {ds.n_trunc} over max_seq_len={args.max_seq_len}); "
        f"tokens p50={L[len(L)//2]} p99={L[int(0.99*len(L))]} max={L[-1]} "
        f"total={sum(L)/1e6:.1f}M",
        flush=True,
    )

    # show one rendered example both ways (template_unreachable guard)
    e = ds[0]
    print("---- rendered example ----", flush=True)
    print(tok.decode(e["input_ids"])[:1200], flush=True)
    print("---- target only ----", flush=True)
    print(tok.decode([t for t in e["labels"] if t != -100])[-400:], flush=True)
    print("--------------------------", flush=True)
    if args.dry_run:
        return

    # fp32 master weights + bf16 autocast: at lr 1e-5 an Adam step is ~1e-3 of a
    # weight, which is below bf16's 2^-8 relative resolution, so pure-bf16
    # parameters would silently round most updates away.
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.float32 if args.master_fp32 else torch.bfloat16,
        attn_implementation=args.attn,
    )
    if hasattr(model.model, "vision_tower"):
        for p in model.model.vision_tower.parameters():
            p.requires_grad_(False)
        for p in model.model.multi_modal_projector.parameters():
            p.requires_grad_(False)
    model.config.use_cache = False

    bds = BatchedDataset(ds, args.token_budget, args.seed)
    sizes = [len(b) for b in bds.batches]
    print(
        f"batches {len(bds)} (token budget {args.token_budget}); "
        f"rows/batch min={min(sizes)} median={sorted(sizes)[len(sizes)//2]} max={max(sizes)}",
        flush=True,
    )

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        optim=args.optim,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=bds,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()

    out = Path(args.out) / "final"
    out.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    trainer.model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    copy_aux_files(Path(args.model), out)
    shutil.copy2(Path(args.model) / "generation_config.json", out / "generation_config.json")
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()
