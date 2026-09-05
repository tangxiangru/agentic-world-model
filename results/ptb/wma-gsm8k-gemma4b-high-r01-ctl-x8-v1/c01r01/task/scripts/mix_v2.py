#!/usr/bin/env python3
"""Mix the RFT rows with a slice of the original SFT corpus.

Continuing training purely on the model's own correct samples narrows the
output distribution; keeping a slice of the exp-02 corpus anchors it.
"""
import argparse
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--rft", default="/home/ben/task/data/rft_v1.jsonl")
ap.add_argument("--sft", default="/home/ben/task/data/sft_v1.jsonl")
ap.add_argument("--n-sft", type=int, default=25000)
ap.add_argument("--out", default="/home/ben/task/data/sft_v2.jsonl")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rng = random.Random(a.seed)
rows = [json.loads(l) for l in open(a.rft)]
n_rft = len(rows)
sft = [json.loads(l) for l in open(a.sft)]
rng.shuffle(sft)
rows += sft[: a.n_sft]
rng.shuffle(rows)
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"rft={n_rft} sft={min(a.n_sft, len(sft))} total={len(rows)} -> {a.out}")
