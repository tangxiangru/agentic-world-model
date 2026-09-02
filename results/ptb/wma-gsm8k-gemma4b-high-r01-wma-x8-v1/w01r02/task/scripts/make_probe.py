#!/usr/bin/env python3
"""Split the official gsm8k TRAIN split into a held-out probe set and a trainable pool.

The benchmark test split is never touched here. The probe set is our own dev set:
it comes from train, it is excluded from every training file, and it is what
`watch_set` / diagnostics point at (protocol rule 7).
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

ds = load_dataset("openai/gsm8k", "main", split="train")

# The 10 fewshot exemplars the grader injects into every prompt (seed 42, shuffled).
fewshot = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42).select(range(10))
fewshot_q = {r["question"] for r in fewshot}

rows = []
for i, r in enumerate(ds):
    ans = r["answer"]
    final = ans.split("####")[-1].strip().replace(",", "")
    rows.append(
        {
            "id": f"train-{i:05d}",
            "question": r["question"],
            "answer": ans,
            "gold": final,
            "is_fewshot": r["question"] in fewshot_q,
        }
    )

rng = random.Random(0)
idx = list(range(len(rows)))
rng.shuffle(idx)
probe_idx = set(i for i in idx if not rows[i]["is_fewshot"])
probe_idx = set(list(sorted(probe_idx, key=lambda i: idx.index(i)))[:300])

probe = [rows[i] for i in sorted(probe_idx)]
pool = [rows[i] for i in range(len(rows)) if i not in probe_idx]

with open(OUT / "probe300.jsonl", "w") as f:
    for r in probe:
        f.write(json.dumps({"id": r["id"], "question": r["question"], "gold": r["gold"]}) + "\n")

with open(OUT / "gsm8k_train_pool.jsonl", "w") as f:
    for r in pool:
        f.write(json.dumps(r) + "\n")

probe_q = {r["question"] for r in probe}
with open(OUT / "probe_questions.json", "w") as f:
    json.dump(sorted(probe_q), f)

print(f"probe300 {len(probe)}  pool {len(pool)}  fewshot_in_pool "
      f"{sum(1 for r in pool if r['is_fewshot'])}")
