"""Build a held-out probe/dev set from the GSM8K TRAIN split.

These items are reserved: they are excluded from every training file this
session builds, so they can be used as a cheap local probe without touching
the benchmark test split.
"""
import json
from datasets import load_dataset

HOLDOUT_START = 7173  # last 300 rows of the 7473-row train split

d = load_dataset("openai/gsm8k", "main")["train"]
rows = []
for i in range(HOLDOUT_START, len(d)):
    r = d[i]
    gold = r["answer"].split("####")[-1].strip().replace(",", "")
    rows.append({"id": f"train-{i}", "question": r["question"], "gold": gold})

with open("data/probe300.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

with open("analysis/watch100.jsonl", "w") as f:
    for r in rows[:100]:
        f.write(json.dumps(r) + "\n")

print(len(rows), "probe rows;", 100, "watch rows")
