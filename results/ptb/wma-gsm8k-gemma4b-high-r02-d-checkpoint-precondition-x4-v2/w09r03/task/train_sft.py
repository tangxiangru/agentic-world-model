"""Completion-only SFT for gemma-3-4b-pt on pre-rendered gemma3-template strings.

The jsonl rows already contain the exact chat string the grader's vLLM sees
(see gsm_format.py / verify_render.py), so nothing here re-templates anything:
tokenize prompt and completion with add_special_tokens=False, mask the prompt,
train. The vision tower and multimodal projector are frozen; the full
Gemma3ForConditionalGeneration is saved so the grader loads it exactly like the
base snapshot.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
# files vLLM/transformers need beside the weights that save_pretrained does not write
EXTRA_FILES = ["preprocessor_config.json", "processor_config.json"]


class PackedRows(Dataset):
    def __init__(self, path, tok, max_seq_len, limit=None):
        self.rows = []
        n_trunc = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                d = json.loads(line)
                p = tok(d["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(d["completion"], add_special_tokens=False)["input_ids"]
                if len(p) + len(c) > max_seq_len:
                    n_trunc += 1
                    continue
                self.rows.append((p, c))
        self.n_dropped = n_trunc
        lens = sorted(len(p) + len(c) for p, c in self.rows)
        self.stats = {
            "n": len(self.rows),
            "dropped_too_long": n_trunc,
            "p50": lens[len(lens) // 2],
            "p99": lens[int(len(lens) * 0.99)],
            "max": lens[-1],
            "total_tokens": sum(lens),
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        ids = p + c
        return {
            "input_ids": ids,
            "labels": [-100] * len(p) + c,
            "length": len(ids),
        }


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids, lab, att = [], [], []
        for b in batch:
            k = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * k)
            lab.append(b["labels"] + [-100] * k)
            att.append([1] * len(b["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(lab),
            "attention_mask": torch.tensor(att),
        }


def _chunk_ce(h, labels, weight):
    return F.cross_entropy(F.linear(h, weight).float(), labels, reduction="sum")


class MaskedLossTrainer(Trainer):
    """Loss over completion tokens only, with the lm_head applied in chunks.

    Gemma3's own forward materialises [B, T, 262208] logits and accelerate then
    upcasts them to fp32, which OOMs at 80 GB for a 16x2400 batch. Here the
    prompt positions are dropped *before* the head runs, so only the ~150
    completion tokens per row ever become logits, and each 2048-token chunk is
    recomputed in backward instead of being kept.
    """

    CHUNK = 2048

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        m = model.module if hasattr(model, "module") else model
        hidden = m.model.language_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).last_hidden_state
        h = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
        labels = inputs["labels"][:, 1:].reshape(-1)
        keep = labels != -100
        h, labels = h[keep], labels[keep]
        n = labels.numel()
        loss = h.new_zeros((), dtype=torch.float32)
        for i in range(0, n, self.CHUNK):
            loss = loss + checkpoint(
                _chunk_ce, h[i : i + self.CHUNK], labels[i : i + self.CHUNK],
                m.lm_head.weight, use_reentrant=False,
            )
        loss = loss / max(n, 1)
        return (loss, None) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_train.jsonl")
    ap.add_argument("--parent", default=SNAP)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--no-gc", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAP)
    ds = PackedRows(args.data, tok, args.max_seq_len, args.limit)
    print("dataset:", json.dumps(ds.stats), flush=True)
    assert ds.n_dropped / max(1, ds.n_dropped + len(ds)) < 0.02, "more than 2% of rows truncate"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.model.vision_tower.requires_grad_(False)
    model.model.multi_modal_projector.requires_grad_(False)
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        bf16=True,
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=3,
        gradient_checkpointing=not args.no_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
    )
    trainer = MaskedLossTrainer(model=model, args=targs, train_dataset=ds, data_collator=Collator(tok.pad_token_id))
    res = trainer.train()
    print("train result:", res, flush=True)

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    for f in EXTRA_FILES:
        src = os.path.join(SNAP, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, f))
    with open(os.path.join(final, "train_stats.json"), "w") as f:
        json.dump({"data": ds.stats, "metrics": res.metrics, "args": vars(args)}, f, indent=2)
    print("saved:", final, flush=True)


if __name__ == "__main__":
    main()
