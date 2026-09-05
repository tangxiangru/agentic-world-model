"""Completion-only SFT for gemma-3-4b-pt on grader-rendered math CoT.

Rows are pre-rendered {prompt, completion} strings (scripts/build_data.py) that
already contain the grader's chat template, so tokenization is
add_special_tokens=False on both halves and the prompt half is masked out.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
from torch.utils.data import Dataset, Sampler

sys.path.insert(0, "/home/ben/task/scripts")
from common import SNAPSHOT, get_tokenizer  # noqa: E402


class Rows(Dataset):
    def __init__(self, recs):
        self.recs = recs

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        return self.recs[i]


class Collator:
    def __init__(self, pad_id: int, pad_to_multiple_of: int = 16):
        self.pad_id = pad_id
        self.mult = pad_to_multiple_of

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        n = ((n + self.mult - 1) // self.mult) * self.mult
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


class TokenBudgetBatches(Sampler):
    """Length-grouped batches with a fixed padded-token budget per micro-batch.

    Fixed row counts waste the GPU here: the corpus mixes ~360-token zero-shot
    rows with ~2400-token few-shot rows. Batches are built once (so
    len(dataloader) is stable across epochs, which the LR schedule relies on)
    and only their order is reshuffled per epoch.
    """

    def __init__(self, lengths, budget: int, seed: int = 0, megabatch: int = 4096):
        self.budget = budget
        g = random.Random(seed)
        idx = list(range(len(lengths)))
        g.shuffle(idx)
        batches = []
        for i in range(0, len(idx), megabatch):
            chunk = sorted(idx[i:i + megabatch], key=lambda j: lengths[j])
            cur, curmax = [], 0
            for j in chunk:
                m = max(curmax, lengths[j])
                if cur and m * (len(cur) + 1) > budget:
                    batches.append(cur)
                    cur, curmax = [j], lengths[j]
                else:
                    cur.append(j)
                    curmax = m
            if cur:
                batches.append(cur)
        g.shuffle(batches)
        self.batches = batches
        self.g = g

    def __iter__(self):
        order = list(range(len(self.batches)))
        self.g.shuffle(order)
        return iter([self.batches[i] for i in order])

    def __len__(self):
        return len(self.batches)


def tokenize(path, tok, max_seq_len, limit=0):
    recs, n_trunc, n_tok, n_sup = [], 0, 0, 0
    with open(path) as f:
        lines = f.readlines()
    if limit:
        lines = lines[:limit]
    prompts = [json.loads(x) for x in lines]
    p_ids = tok([r["prompt"] for r in prompts], add_special_tokens=False)["input_ids"]
    c_ids = tok([r["completion"] for r in prompts], add_special_tokens=False)["input_ids"]
    for p, c in zip(p_ids, c_ids):
        if len(p) + len(c) > max_seq_len:
            n_trunc += 1
            continue
        recs.append({"input_ids": p + c, "labels": [-100] * len(p) + c,
                     "length": len(p) + len(c)})
        n_tok += len(p) + len(c)
        n_sup += len(c)
    print(f"[data] {len(recs)} rows, dropped {n_trunc} over {max_seq_len} tok "
          f"({n_trunc / max(1, len(lines)):.3%}); {n_tok/1e6:.1f}M tokens, "
          f"{n_sup/1e6:.1f}M supervised", flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/ben/task/data/sft_exp02.jsonl")
    ap.add_argument("--model", default=SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-strategy", default="epoch")
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--min-lr-ratio", type=float, default=0.0)
    ap.add_argument("--liger", type=int, default=1)
    ap.add_argument("--token-budget", type=int, default=24576)
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, Trainer, TrainingArguments)

    tok = get_tokenizer()
    recs = tokenize(args.data, tok, args.max_seq_len, args.limit)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.config.use_cache = False

    frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            frozen += p.numel()
    train_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {type(model).__name__} trainable {train_n/1e9:.2f}B "
          f"frozen {frozen/1e9:.2f}B", flush=True)

    kw = {}
    if args.scheduler == "cosine_with_min_lr":
        kw["lr_scheduler_kwargs"] = {"min_lr_rate": args.min_lr_ratio}

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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy=args.save_strategy,
        save_total_limit=4,
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        optim=args.optim,
        dataloader_num_workers=4,
        use_liger_kernel=bool(args.liger),
        **kw,
    )

    class BudgetTrainer(Trainer):
        def get_train_dataloader(self):
            from torch.utils.data import DataLoader
            sampler = TokenBudgetBatches([r["length"] for r in recs],
                                         args.token_budget, seed=args.seed)
            print(f"[sampler] {len(sampler)} micro-batches/epoch, "
                  f"mean {len(recs)/len(sampler):.1f} rows", flush=True)
            return self.accelerator.prepare(DataLoader(
                self.train_dataset, batch_sampler=sampler,
                collate_fn=self.data_collator, num_workers=4, pin_memory=True))

    T = BudgetTrainer if args.token_budget else Trainer
    trainer = T(model=model, args=targs, train_dataset=Rows(recs),
                data_collator=Collator(tok.pad_token_id))
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok_plain = __import__("transformers").AutoTokenizer.from_pretrained(SNAPSHOT)
    tok_plain.save_pretrained(final)
    for extra in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(SNAPSHOT, extra)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, os.path.join(final, extra))
    print(f"[done] saved {final}", flush=True)


if __name__ == "__main__":
    main()
