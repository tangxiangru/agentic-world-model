#!/usr/bin/env python3
"""Third-stage data: self-generated correct chains (RFT) mixed with an equal
slice of unseen teacher rows as a regulariser."""
import json
import random
import sys

rft = [json.loads(l) for l in open("data/rft_v1.jsonl")]
pool = [json.loads(l) for l in open("data/train_v2.jsonl")][32000:]  # unused by exp-04
n_teacher = min(len(rft), len(pool))
rng = random.Random(3)
rng.shuffle(pool)
rows = rft + pool[:n_teacher]
rng.shuffle(rows)

bad = [r for r in rows if not r["target"].endswith("<end_of_turn>")
       or r["target"].count("ANSWER:") != 1]
assert not bad, f"{len(bad)} malformed targets"

with open("data/train_v3.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
with open("data/train_v3_check.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")
print(f"rft rows {len(rft)}  teacher rows {n_teacher}  total {len(rows)}")
