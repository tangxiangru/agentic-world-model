#!/usr/bin/env python3
"""Attach the grader's exact 10-shot system block to a fraction of SFT rows.

inspect_evals/gsm8k builds its system message from the GSM8K TRAIN split
(hf_dataset shuffle seed 42, limit 10) and passes it to the model on every
item. A model trained only zero-shot answers in a different style under that
prefix; this makes a slice of training rows carry it verbatim.
"""
from __future__ import annotations

import argparse
import json
import random


def eval_system_message() -> str:
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

    fewshots = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=42,
        limit=10,
    )
    return "\n\n".join(sample_to_fewshot(s) for s in fewshots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sysmsg = eval_system_message()
    print(f"system message: {len(sysmsg)} chars")

    rows = [json.loads(l) for l in open(a.inp)]
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    if a.n:
        rows = rows[: a.n]
    k = int(len(rows) * a.frac)
    for i, r in enumerate(rows):
        if i < k:
            r["system"] = sysmsg
    rng.shuffle(rows)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows ({k} with the 10-shot prefix) to {a.out}")


if __name__ == "__main__":
    main()
