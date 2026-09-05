"""Unique GSM8K-train-derived problems with their gold answers, for RFT sampling.

Same sources and same holdout exclusion as scripts/build_data.py; the gsm8k TEST
split is never read.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

import pyarrow.parquet as pq
from datasets import load_dataset

OMI2_DIR = (
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
    "469216e3f46f4dacf476b382e192485ea51a143e/data/"
)
INT_RE = re.compile(r"^-?\d+(\.\d+)?$")


def norm(p: str) -> str:
    return re.sub(r"\s+", " ", p.strip().lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/rft_problems.jsonl")
    ap.add_argument("--n-augmented", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    holdout = {
        norm(json.loads(l)["question"])
        for l in open("/home/ben/task/data/dev_train.jsonl")
    }
    rng = random.Random(args.seed)

    seen: set[str] = set()
    orig, aug = [], []
    gsm = load_dataset("openai/gsm8k", "main")["train"]
    for r in gsm:
        k = norm(r["question"])
        if k in holdout or k in seen:
            continue
        seen.add(k)
        orig.append(
            {
                "question": r["question"].strip(),
                "gold": r["answer"].split("####")[-1].strip().replace(",", ""),
                "src": "gsm8k_train",
            }
        )

    for i in range(3):
        pf = pq.ParquetFile(OMI2_DIR + f"train_1M-0000{i}-of-00003.parquet")
        for b in pf.iter_batches(batch_size=50000):
            for r in b.to_pylist():
                if r["problem_source"] != "augmented_gsm8k":
                    continue
                ans = (r["expected_answer"] or "").strip().replace(",", "")
                if not INT_RE.match(ans):
                    continue
                k = norm(r["problem"])
                if k in holdout or k in seen:
                    continue
                seen.add(k)
                aug.append(
                    {"question": r["problem"].strip(), "gold": ans, "src": "augmented"}
                )

    rng.shuffle(aug)
    rows = orig + aug[: args.n_augmented]
    rng.shuffle(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} problems -> {args.out} ({len(orig)} gsm8k_train, {len(aug[:args.n_augmented])} augmented)")


if __name__ == "__main__":
    main()
