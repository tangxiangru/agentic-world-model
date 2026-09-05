#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on prompt/completion JSONL.

Rows are pre-rendered strings (see build_sft_data.py): the prompt is the exact
gemma3.jinja render the grader produces, the completion ends with <end_of_turn>.
Loss is taken on completion tokens only. The vision tower and the multimodal
projector are frozen (no images in this corpus) so their optimizer state costs
nothing; the checkpoint keeps the full Gemma3ForConditionalGeneration shape so
vLLM loads it exactly like the base snapshot.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
IGNORE = -100


class PromptCompletionDataset(Dataset):
    def __init__(self, path, tok, max_seq_len, limit=None):
        self.rows = []
        n_trunc = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
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
        labels = [IGNORE] * len(p) + list(c)
        return {"input_ids": ids, "labels": labels}


class TokenBudgetBatches:
    """Length-bucketed batches with a fixed token budget per micro-batch.

    Gemma-3's 262k vocab makes the logits tensor the memory bottleneck, so what
    must be held constant across micro-batches is tokens, not rows. Batches are
    computed once (deterministically); only their order is reshuffled per epoch.
    """

    def __init__(self, lengths, budget, seed=0, mega=2048):
        rng = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(lengths), generator=rng).tolist()
        self.batches = []
        for s in range(0, len(order), mega):
            chunk = sorted(order[s : s + mega], key=lambda i: lengths[i])
            cur, cur_max = [], 0
            for i in chunk:
                m = max(cur_max, lengths[i])
                if cur and m * (len(cur) + 1) > budget:
                    self.batches.append(cur)
                    cur, cur_max = [i], lengths[i]
                else:
                    cur, cur_max = cur + [i], m
            if cur:
                self.batches.append(cur)
        self.epoch = 0

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        rng = torch.Generator().manual_seed(1234 + self.epoch)
        self.epoch += 1
        for k in torch.randperm(len(self.batches), generator=rng).tolist():
            yield self.batches[k]


@dataclass
class Collator:
    pad_id: int

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            k = m - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [IGNORE] * k)
            attn.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parent", default=SNAP)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=2816)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--token-budget", type=int, default=8192,
                    help="max padded tokens per micro-batch (memory is vocab-logit bound)")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--save-total-limit", type=int, default=6)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAP)
    ds = PromptCompletionDataset(args.data, tok, args.max_seq_len, args.limit)
    print(f"rows={len(ds)} dropped_too_long={ds.n_trunc} "
          f"({ds.n_trunc / max(1, len(ds) + ds.n_trunc):.3%}) "
          f"tokens={sum(ds.lengths):,} maxlen={max(ds.lengths)}", flush=True)
    if args.dry_run:
        p, c = ds.rows[0]
        print("--- example prompt ---\n" + tok.decode(p))
        print("--- example completion ---\n" + tok.decode(c))
        return

    cfg = AutoConfig.from_pretrained(args.parent)
    model = AutoModelForCausalLM.from_pretrained(
        args.parent, config=cfg, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"frozen vision params: {n_frozen/1e6:.0f}M; trainable: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e9:.2f}B", flush=True)
    model.config.use_cache = False

    batcher = TokenBudgetBatches(ds.lengths, args.token_budget, seed=args.seed)
    steps_per_epoch = math.ceil(len(batcher) / args.grad_accum)
    print(f"micro-batches/epoch={len(batcher)} "
          f"(rows/mb p50={sorted(len(b) for b in batcher.batches)[len(batcher)//2]})", flush=True)
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=4,
        max_grad_norm=1.0,
    )
    print(f"steps/epoch={steps_per_epoch} total_steps={int(steps_per_epoch*args.epochs)}", flush=True)

    class T(Trainer):
        def get_train_dataloader(self):
            from torch.utils.data import DataLoader
            return DataLoader(
                ds,
                batch_sampler=batcher,
                collate_fn=Collator(tok.pad_token_id or 0),
                num_workers=self.args.dataloader_num_workers,
                pin_memory=True,
            )

    trainer = T(model=model, args=targs, train_dataset=ds,
                data_collator=Collator(tok.pad_token_id or 0))
    trainer.train()
    trainer.save_model(os.path.join(args.out, "final"))
    tok.save_pretrained(os.path.join(args.out, "final"))
    # keep the processor bits so vLLM loads the multimodal config unchanged
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(SNAP).save_pretrained(os.path.join(args.out, "final"))
    except Exception as e:  # noqa: BLE001
        print("processor save skipped:", e)
    with open(os.path.join(args.out, "final", "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print("peak GPU mem GB:", torch.cuda.max_memory_allocated()/1e9)
    print("saved", os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
