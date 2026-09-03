#!/usr/bin/env python3
"""Build a held-out probe/dev split from the GSM8K *train* split.

Nothing from the benchmark test split is touched here. The items written to
data/dev_train_holdout.jsonl are removed from every training file later on, so
they stay an honest local probe.
"""
import json
import random

from datasets import load_dataset

N_DEV = 300
SEED = 12345

ds = load_dataset("openai/gsm8k", "main")["train"]
idx = list(range(len(ds)))
random.Random(SEED).shuffle(idx)
dev_idx = sorted(idx[:N_DEV])

with open("data/dev_train_holdout.jsonl", "w") as f:
    for i in dev_idx:
        r = ds[i]
        gold = r["answer"].split("####")[-1].strip()
        f.write(json.dumps({"id": f"train-{i}", "question": r["question"],
                            "gold": gold, "reasoning": r["answer"].split("####")[0].strip()}) + "\n")

with open("data/dev_holdout_idx.json", "w") as f:
    json.dump(dev_idx, f)

print("wrote", len(dev_idx), "dev items")
