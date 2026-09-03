#!/usr/bin/env python3
"""Split GSM8K *train* into an SFT pool and a held-out dev set.

The benchmark test split is never touched here: it is the grader's, and rule 7
allows it only as input to the contamination checker.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

ds = load_dataset("openai/gsm8k", "main", split="train")
idx = list(range(len(ds)))
random.Random(0).shuffle(idx)

dev_idx = idx[:300]
pool_idx = idx[300:]


def gold(ans: str) -> str:
    return ans.split("####")[-1].strip().replace(",", "")


with (OUT / "dev_heldout300.jsonl").open("w") as f:
    for i in dev_idx:
        r = ds[i]
        f.write(json.dumps({"id": f"trdev-{i}", "question": r["question"], "gold": gold(r["answer"])}) + "\n")

with (OUT / "watch40.jsonl").open("w") as f:
    for i in dev_idx[:40]:
        r = ds[i]
        f.write(json.dumps({"id": f"trdev-{i}", "question": r["question"], "gold": gold(r["answer"])}) + "\n")

with (OUT / "train_pool.jsonl").open("w") as f:
    for i in pool_idx:
        r = ds[i]
        f.write(json.dumps({"id": f"tr-{i}", "question": r["question"], "answer": r["answer"], "gold": gold(r["answer"])}) + "\n")

print("dev", len(dev_idx), "pool", len(pool_idx))
