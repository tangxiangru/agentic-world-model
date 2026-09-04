"""Full-parameter SFT of gemma-3-4b-pt, token-budget batched.

Same data contract as scripts/train_sft.py (pre-rendered prompt/completion, loss
on the completion only), but micro-batches are built to a fixed *token* budget
instead of a fixed example count.  exp-02 showed the fixed-count version runs at
~176 TFLOPs on 330-token rows and ~470 TFLOPs on 2400-token rows: with 88% short
rows the run was leaving ~3x on the table.  A token budget keeps every
micro-batch the same size in tokens, which is also what bounds memory (the
262144-wide lm_head logits dominate).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def build_microbatches(rows, budget, seed):
    """rows: list of (prompt_ids, completion_ids). Returns list of index lists."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]) + len(rows[i][1]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        n = len(rows[i][0]) + len(rows[i][1])
        m = max(cur_max, n)
        if cur and m * (len(cur) + 1) > budget:
            batches.append(cur)
            cur, cur_max = [i], n
        else:
            cur.append(i)
            cur_max = m
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    return batches


class BatchedRows(Dataset):
    """Each item is one ready-made micro-batch."""

    def __init__(self, path, tok, max_seq_len, budget, seed, limit=None):
        rows = []
        n_drop = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                if len(p) + len(c) > max_seq_len:
                    n_drop += 1
                    continue
                rows.append((p, c))
        lens = sorted(len(p) + len(c) for p, c in rows)
        self.rows = rows
        self.batches = build_microbatches(rows, budget, seed)
        self.total_tokens = sum(lens)
        sizes = sorted(len(b) for b in self.batches)
        print(
            f"[data] kept {len(rows)} rows, dropped {n_drop} over max_seq_len={max_seq_len} "
            f"({n_drop / max(1, n_drop + len(rows)):.2%}); len p50={lens[len(lens) // 2]} "
            f"p99={lens[int(0.99 * (len(lens) - 1))]} max={lens[-1]}; "
            f"{len(self.batches)} micro-batches, size p50={sizes[len(sizes) // 2]} "
            f"min={sizes[0]} max={sizes[-1]}; {self.total_tokens / 1e6:.1f}M tokens",
            flush=True,
        )

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, b):
        out = []
        for i in self.batches[b]:
            p, c = self.rows[i]
            out.append({"input_ids": p + c, "labels": [-100] * len(p) + list(c)})
        return out


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, wrapped):
        feats = wrapped[0]  # per_device_train_batch_size is 1; the item IS the batch
        m = max(len(f["input_ids"]) for f in feats)
        ids, labels, mask = [], [], []
        for f in feats:
            k = m - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            mask.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def save_full(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(SNAPSHOT):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        src = os.path.join(SNAPSHOT, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, fn))
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoProcessor.from_pretrained(SNAPSHOT).save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(SNAPSHOT).save_pretrained(out_dir)
    gc = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "cache_implementation": "hybrid",
    }
    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        json.dump(gc, f, indent=2)
    print(f"[save] wrote {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default=SNAPSHOT)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--token-budget", type=int, default=9600)
    ap.add_argument("--lr", type=float, default=1.4e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ds = BatchedRows(args.data, tok, args.max_seq_len, args.token_budget, args.seed, args.limit)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    # a parent checkpoint from an earlier card carries the greedy generation config
    # (temperature 0.0 with do_sample False); transformers refuses to re-serialise it
    # and save_pretrained would die *after* training. Reset it now; save_full writes
    # the greedy config back by hand at the end.
    from transformers import GenerationConfig

    model.generation_config = GenerationConfig.from_pretrained(SNAPSHOT)
    for n, p in model.named_parameters():
        if n.startswith("model.vision_tower") or n.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)
    print(f"[model] trainable {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B", flush=True)

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
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        disable_tqdm=True,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 100000,
        save_total_limit=1,
        save_safetensors=True,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds, data_collator=Collator(tok.pad_token_id or 0)
    )
    t0 = time.time()
    out = trainer.train()
    print(f"[train] {(time.time() - t0) / 60:.1f} min, {out.metrics}", flush=True)
    save_full(model, os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
