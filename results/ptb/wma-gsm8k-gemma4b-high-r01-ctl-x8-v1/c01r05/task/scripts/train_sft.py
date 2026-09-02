"""Completion-only SFT for google/gemma-3-4b-pt on math word problems.

Rows are pre-rendered by scripts/build_sft_data.py as {"prompt", "completion"}
strings that already contain the gemma3 control tokens, so the trainer never
touches a chat template: what is trained is byte-identical to what the grader
renders (verified by scripts/check_template.py).

Loss is on the completion only, including the final <end_of_turn>.
Rows longer than --max-seq-len are dropped, not truncated.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


class PackedRows(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def token_batches(lengths: list[int], budget: int, seed: int) -> list[list[int]]:
    """Length-sorted batches with a fixed *padded* token budget.

    Keeps every micro-batch at roughly `budget` real+pad tokens, so the GPU sees
    a constant amount of work whether the rows are 300 or 2500 tokens long.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lengths[i])
        if cur and m * (len(cur) + 1) > budget:
            batches.append(cur)
            cur, cur_max = [i], lengths[i]
        else:
            cur, cur_max = cur + [i], m
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    return batches


class TokenBudgetTrainer(Trainer):
    """Trainer that batches by token budget instead of a fixed example count."""

    def set_batches(self, batches, collator, workers):
        self._batches = batches
        self._collator = collator
        self._workers = workers

    def get_train_dataloader(self):  # type: ignore[override]
        from torch.utils.data import DataLoader

        dl = DataLoader(
            self.train_dataset,
            batch_sampler=self._batches,
            collate_fn=self._collator,
            num_workers=self._workers,
            pin_memory=True,
        )
        return self.accelerator.prepare(dl)


@dataclass
class Collator:
    pad_id: int

    def __call__(self, feats: list[dict]) -> dict:
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


def tokenize(path: str, tok, max_len: int, limit: int | None, seed: int) -> list[dict]:
    raw = []
    with open(path) as fh:
        for line in fh:
            raw.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(raw)
    if limit:
        raw = raw[:limit]

    prompts = [r["prompt"] for r in raw]
    comps = [r["completion"] for r in raw]
    p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
    c_ids = tok(comps, add_special_tokens=False)["input_ids"]

    rows, dropped, lens = [], 0, []
    for p, c in zip(p_ids, c_ids):
        if len(p) + len(c) > max_len:
            dropped += 1
            continue
        rows.append({"input_ids": p + c, "labels": [-100] * len(p) + c})
        lens.append(len(p) + len(c))
    lens.sort()
    print(
        f"[data] {path}: kept {len(rows)} / {len(raw)}  dropped(too long) {dropped} "
        f"({dropped / max(1, len(raw)):.2%})",
        flush=True,
    )
    if lens:
        print(
            f"[data] token length p50={lens[len(lens) // 2]} p90={lens[int(len(lens) * 0.9)]} "
            f"max={lens[-1]}  total={sum(lens) / 1e6:.1f}M",
            flush=True,
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default=SNAP)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--token-budget", type=int, default=32768)
    ap.add_argument("--liger", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAP)
    pad_id = tok.pad_token_id or 0

    rows = tokenize(args.data, tok, args.max_seq_len, args.limit, args.seed)

    # --- self-checks that would otherwise be a clean-looking wrong answer ---
    eot = tok.convert_tokens_to_ids(common.EOT)
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    bad_eos = sum(1 for r in rows if eot not in r["input_ids"][-3:])
    zero_loss = sum(1 for r in rows if all(x == -100 for x in r["labels"]))
    print(f"[check] rows whose tail lacks {common.EOT}: {bad_eos}", flush=True)
    print(f"[check] rows with zero loss tokens: {zero_loss}", flush=True)
    assert bad_eos == 0 and zero_loss == 0, "target/eos check failed"
    print("[check] decoded tail of row 0:", repr(tok.decode(rows[0]["input_ids"][-40:])), flush=True)
    print("[check] loss span of row 0:", repr(tok.decode([t for t in rows[0]["labels"] if t != -100])[:200]), flush=True)
    if args.dry_run:
        return

    print("[model] loading", args.parent, flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent,
        dtype=torch.float32,
        attn_implementation=args.attn,
    )
    # text-only training: the vision tower never sees a gradient
    frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {trainable / 1e9:.2f}B, frozen {frozen / 1e9:.2f}B", flush=True)
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        optim=args.optim,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        use_liger_kernel=args.liger,
    )

    lengths = [len(r["input_ids"]) for r in rows]
    batches = token_batches(lengths, args.token_budget, args.seed)
    tot = sum(lengths)
    pad = sum(max(lengths[i] for i in b) * len(b) for b in batches)
    print(
        f"[batches] {len(batches)} micro-batches, {tot / 1e6:.1f}M real tokens, "
        f"{pad / 1e6:.1f}M padded ({1 - tot / pad:.1%} pad), "
        f"{len(batches) / args.accum:.0f} optimizer steps/epoch",
        flush=True,
    )

    trainer = TokenBudgetTrainer(
        model=model,
        args=targs,
        train_dataset=PackedRows(rows),
        data_collator=Collator(pad_id),
    )
    trainer.set_batches(batches, Collator(pad_id), targs.dataloader_num_workers)
    trainer.train()

    print("[save] merging to", args.out, flush=True)
    final = Path(args.out) / "final"
    model.config.use_cache = True
    trainer.model.to(torch.bfloat16)
    trainer.model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    print("[save] done", flush=True)


if __name__ == "__main__":
    main()
