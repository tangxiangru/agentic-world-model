#!/usr/bin/env python3
"""Record WHICH benchmark items the evaluation protocol scores.

This is a bookkeeping artefact, not training data. It writes only the stable
sample id and the gold numeric answer for the first --limit items of the
inspect_evals/gsm8k test split, in dataset order, so that later cards can
(a) verify they scored the same items and (b) select watch sets by id.

Questions and reference solutions are deliberately NOT written out: no test
item may reach training data in any form (task rules 3-4, protocol rule 7).
"""
from __future__ import annotations

import argparse
import json

from datasets import load_dataset
from inspect_evals.utils import create_stable_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--out", default="/home/ben/task/data/exp-01_eval_manifest.jsonl")
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split="test")
    with open(args.out, "w") as f:
        for rec in ds.select(range(args.limit)):
            gold = rec["answer"].split("####")[-1].strip()
            sid = create_stable_id(rec["question"], prefix="gsm8k")
            f.write(json.dumps({"id": sid, "gold": gold, "note": "eval item; never training data"}) + "\n")
    print(f"wrote {args.limit} rows to {args.out}")


if __name__ == "__main__":
    main()
