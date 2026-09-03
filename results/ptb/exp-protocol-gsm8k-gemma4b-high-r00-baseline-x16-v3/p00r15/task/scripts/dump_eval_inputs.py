#!/usr/bin/env python3
"""Dump the exact items the eval protocol scores, so a card's protocol is reproducible.

This reads the official inspect_evals/gsm8k test split the same way evaluate.py does
(no shuffle, first --limit items) and writes {id, question, gold} jsonl. The dump is a
record of WHICH items were scored; it is never used as training data.
"""
from __future__ import annotations

import argparse
import json

from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import record_to_sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    ds = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="test",
        sample_fields=record_to_sample,
    )
    with open(args.out, "w") as f:
        for sample in list(ds)[: args.limit]:
            f.write(json.dumps({"id": sample.id, "question": sample.input, "gold": sample.target}) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
