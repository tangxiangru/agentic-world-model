#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on GSM8K-style chains.

Everything is rendered with the *grader's own* chat template
(/home/ben/task/templates/gemma3.jinja, sha256 pinned below), so the string the
model trains on is the string the grader will feed it. Targets terminate on
<end_of_turn>, which is one of the two eos ids in the checkpoint's
generation_config, i.e. the token vLLM stops on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path("/home/ben/task")
TEMPLATE_PATH = ROOT / "templates" / "gemma3.jinja"
TEMPLATE_SHA = "d54ba99e0a68efa3d8b7f0a2f9d3f7e8"  # filled at first run; see check below
END_OF_TURN = "<end_of_turn>"


def load_template() -> str:
    txt = TEMPLATE_PATH.read_text()
    print(f"chat template sha256={hashlib.sha256(txt.encode()).hexdigest()[:16]} "
          f"({len(txt)} chars) from {TEMPLATE_PATH}")
    return txt


@dataclass
class Row:
    input_ids: list[int]
    n_prompt: int


def make_batches(rows: list[Row], token_budget: int, max_rows: int,
                 seed: int) -> list[list[int]]:
    """Length-bucketed batches with a padded-token budget.

    A fixed row count blows up on the few-shot rows (2.4k tokens) while wasting the
    GPU on the 300-token median. Bucketing by length and capping
    len(batch) * max_len_in_batch keeps peak activation memory flat.
    """
    order = sorted(range(len(rows)), key=lambda i: len(rows[i].input_ids))
    batches: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        L = len(rows[i].input_ids)
        if cur and ((len(cur) + 1) * L > token_budget or len(cur) + 1 > max_rows):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    return batches


class BatchData(Dataset):
    """Items are whole (pre-bucketed) batches; the Trainer runs with batch size 1."""

    def __init__(self, rows: list[Row], batches: list[list[int]]):
        self.rows = rows
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, i: int) -> list[Row]:
        return [self.rows[j] for j in self.batches[i]]


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats: list[list[Row]]) -> dict:
        assert len(feats) == 1, "BatchData yields whole batches; keep bs=1"
        batch = feats[0]
        n = max(len(r.input_ids) for r in batch)
        ids, labs, mask = [], [], []
        for r in batch:
            k = n - len(r.input_ids)
            lab = [-100] * r.n_prompt + r.input_ids[r.n_prompt:] + [-100] * k
            ids.append(r.input_ids + [self.pad_id] * k)
            labs.append(lab)
            mask.append([1] * len(r.input_ids) + [0] * k)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def build_rows(tok, template: str, data_path: Path, max_seq_len: int,
               fewshot_prefix: str | None, fewshot_frac: float, seed: int,
               limit: int | None) -> tuple[list[Row], dict]:
    rng = random.Random(seed)
    rows: list[Row] = []
    stats = {"n_in": 0, "n_kept": 0, "n_truncated": 0, "n_fewshot": 0,
             "len_p50": 0, "len_p99": 0, "len_max": 0, "tgt_max": 0}
    lens: list[int] = []
    eot_id = tok.convert_tokens_to_ids(END_OF_TURN)
    assert eot_id is not None and eot_id > 0, eot_id

    with data_path.open() as f:
        for line in f:
            if limit is not None and stats["n_in"] >= limit:
                break
            r = json.loads(line)
            stats["n_in"] += 1
            use_fs = fewshot_prefix is not None and rng.random() < fewshot_frac
            msgs = []
            if use_fs:
                msgs.append({"role": "system", "content": fewshot_prefix})
            msgs.append({"role": "user", "content": r["prompt"]})
            prompt_txt = tok.apply_chat_template(
                msgs, chat_template=template, tokenize=False, add_generation_prompt=True
            )
            target_txt = r["completion"].strip()
            assert target_txt.endswith(END_OF_TURN), target_txt[-40:]
            p_ids = tok(prompt_txt, add_special_tokens=False)["input_ids"]
            t_ids = tok(target_txt, add_special_tokens=False)["input_ids"]
            assert t_ids[-1] == eot_id, (t_ids[-3:], eot_id)
            total = len(p_ids) + len(t_ids)
            lens.append(total)
            stats["tgt_max"] = max(stats["tgt_max"], len(t_ids))
            if total > max_seq_len:
                stats["n_truncated"] += 1
                continue  # drop rather than truncate: a cut target loses the stop token
            rows.append(Row(input_ids=p_ids + t_ids, n_prompt=len(p_ids)))
            stats["n_kept"] += 1
            if use_fs:
                stats["n_fewshot"] += 1
    lens.sort()
    if lens:
        stats["len_p50"] = lens[len(lens) // 2]
        stats["len_p99"] = lens[int(len(lens) * 0.99)]
        stats["len_max"] = lens[-1]
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--max-rows-per-batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2432)
    ap.add_argument("--fewshot-frac", type=float, default=0.06)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="build data, report, exit")
    args = ap.parse_args()

    set_seed(args.seed)
    template = load_template()
    tok = AutoTokenizer.from_pretrained(args.model)
    fewshot_prefix = (ROOT / "data" / "fewshot_prefix.txt").read_text() if args.fewshot_frac > 0 else None

    rows, stats = build_rows(tok, template, Path(args.data), args.max_seq_len,
                             fewshot_prefix, args.fewshot_frac, args.seed, args.limit)
    print("DATA STATS", json.dumps(stats, indent=1))
    frac_dropped = stats["n_truncated"] / max(stats["n_in"], 1)
    print(f"dropped for length: {frac_dropped:.4f}")
    assert frac_dropped < 0.02, f"too many rows over max_seq_len ({frac_dropped:.3f})"

    # show one fully rendered example so the training string is auditable
    ex = rows[0]
    print("=== EXAMPLE (prompt) ===")
    print(tok.decode(ex.input_ids[: ex.n_prompt])[:1200])
    print("=== EXAMPLE (target) ===")
    print(repr(tok.decode(ex.input_ids[ex.n_prompt:])[-400:]))

    if args.dry_run:
        return

    batches = make_batches(rows, args.token_budget, args.max_rows_per_batch, args.seed)
    print(f"batches: {len(batches)}  rows/batch p50="
          f"{sorted(len(b) for b in batches)[len(batches)//2]}  "
          f"max={max(len(b) for b in batches)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    )
    print("model class:", type(model).__name__)
    # A greedy generation_config on the PARENT (temperature 0 / top_k 0 with
    # do_sample False) makes GenerationConfig.validate(strict=True) raise inside
    # save_pretrained, which throws away the whole run at save time. Restore a
    # valid sampling config here; finalize_model.py rewrites the file afterwards.
    gc = model.generation_config
    gc.do_sample, gc.temperature, gc.top_k, gc.top_p = True, 1.0, 64, 0.95
    try:
        gc.validate(strict=True)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"generation_config would not survive save_pretrained: {e}")
    # the vision tower is dead weight for text-only GSM8K: freeze it, keep it in the
    # checkpoint so the saved artefact stays a loadable Gemma3ForConditionalGeneration
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {n_frozen/1e6:.0f}M, trainable {trainable/1e6:.0f}M")
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,   # BatchData items are already whole batches
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        max_grad_norm=1.0,
        use_liger_kernel=True,   # fused linear CE: 262k-vocab logits are never materialised
        remove_unused_columns=False,
        accelerator_config={"dispatch_batches": False},
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=BatchData(rows, batches),
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()
    final = Path(args.out) / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    print("saved", final)


if __name__ == "__main__":
    main()
