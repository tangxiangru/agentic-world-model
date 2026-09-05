#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on prompt/completion jsonl.

Strings are pre-rendered by build_data.py with scripts/fmt.py, i.e. with the
same templates/gemma3.jinja the grader hands to `vllm serve`. This script only
tokenizes them, so training and grading cannot drift apart.

Rows longer than --max-seq-len are DROPPED, never truncated: under
completion-only loss a truncated row silently carries zero loss tokens
(pitfall seq_len_truncation).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

END_OF_TURN_ID = 106


class SFTRows(Dataset):
    def __init__(self, path, tokenizer, max_seq_len, limit=None):
        self.rows = []
        n_drop = 0
        lens = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                e = json.loads(line)
                p = tokenizer(e["prompt"], add_special_tokens=False)["input_ids"]
                c = tokenizer(e["completion"], add_special_tokens=False)["input_ids"]
                if c[-1] != END_OF_TURN_ID:
                    raise ValueError(
                        f"row {i}: completion does not end with <end_of_turn> "
                        f"(got id {c[-1]})"
                    )
                if len(p) + len(c) > max_seq_len:
                    n_drop += 1
                    continue
                lens.append(len(p) + len(c))
                self.rows.append((p, c))
        lens.sort()
        self.stats = {
            "kept": len(self.rows),
            "dropped": n_drop,
            "drop_frac": n_drop / max(1, n_drop + len(self.rows)),
            "p50": lens[len(lens) // 2] if lens else 0,
            "p99": lens[int(len(lens) * 0.99)] if lens else 0,
            "max": lens[-1] if lens else 0,
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        ids = p + c
        labels = [-100] * len(p) + list(c)
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


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


class ChunkedCETrainer(Trainer):
    """Cross-entropy over supervised positions only, in checkpointed chunks.

    Gemma-3's vocabulary is 262144, so the stock forward materialises an fp32
    logit tensor of B*T*262144*4 bytes (23 GB for 8x2816) and OOMs on an H100.
    We take hidden states, gather only the positions whose label != -100
    (completion-only loss makes that a minority of tokens), and run
    lm_head + CE in chunks under torch.utils.checkpoint, so peak logit memory
    is one chunk instead of the whole batch.
    """

    chunk = 1024

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        base = model.module if hasattr(model, "module") else model
        out = base.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        hidden = out[0]
        sel_h = hidden[:, :-1, :]
        sel_y = labels[:, 1:]
        mask = sel_y != -100
        h = sel_h[mask]
        y = sel_y[mask]
        n = h.shape[0]

        def block(hh, yy):
            lg = base.lm_head(hh).float()
            return torch.nn.functional.cross_entropy(lg, yy, reduction="sum")

        total = None
        for i in range(0, n, self.chunk):
            part = torch.utils.checkpoint.checkpoint(
                block, h[i : i + self.chunk], y[i : i + self.chunk], use_reentrant=False
            )
            total = part if total is None else total + part
        loss = total / max(1, n)
        return (loss, out) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2816)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = SFTRows(args.data, tok, args.max_seq_len, args.limit)
    print("data stats:", json.dumps(ds.stats), flush=True)
    if ds.stats["drop_frac"] > 0.02:
        raise SystemExit(
            f"ABORT: {ds.stats['drop_frac']:.3%} of rows exceed max_seq_len "
            f"={args.max_seq_len} (max row {ds.stats['max']}). Raise it."
        )
    if args.dry_run:
        p, c = ds.rows[0]
        print("--- decoded prompt tail ---")
        print(repr(tok.decode(p[-120:])))
        print("--- decoded completion ---")
        print(repr(tok.decode(c)))
        print("--- last 3 completion ids ---", c[-3:])
        return

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False
    # train the language model only; the vision tower is dead weight for gsm8k
    # but must stay in the checkpoint for vLLM to load Gemma3ForConditionalGeneration
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_train/1e9:.2f}B  frozen {n_frozen/1e9:.2f}B", flush=True)

    # A parent checkpoint that has already been through scripts/set_decode.py
    # carries do_sample=False together with temperature=0.0 / top_k=-1.
    # GenerationConfig.validate() rejects that combination, and save_pretrained
    # raises on it -- which killed exp-05 at step 1000 with the weights unwritten.
    # Restore the base model's sampling-shaped config for training; set_decode.py
    # is re-applied to the output directory afterwards.
    gc = model.generation_config
    if getattr(gc, "do_sample", True) is False:
        gc.do_sample = True
        gc.temperature = 1.0
        gc.top_k = 64
        gc.top_p = 0.95
        print("sanitised parent generation_config for saving", flush=True)
    gc.validate()

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=3,
        save_only_model=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        group_by_length=True,
        length_column_name="length",
        report_to=[],
        seed=args.seed,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
    )

    trainer = ChunkedCETrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    try:
        AutoProcessor.from_pretrained(args.model).save_pretrained(final)
    except Exception as e:  # noqa: BLE001
        print("processor save skipped:", e)
    print("saved", final)


if __name__ == "__main__":
    main()
