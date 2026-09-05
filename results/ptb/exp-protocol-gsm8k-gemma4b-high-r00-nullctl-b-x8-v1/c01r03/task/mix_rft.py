#!/usr/bin/env python3
"""Mix the on-policy RFT data with an anchor slice of the stage-1 SFT data."""
import argparse
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--rft", default="data/rft.jsonl")
ap.add_argument("--anchor", default="data/sft.jsonl")
ap.add_argument("--n-anchor", type=int, default=25000)
ap.add_argument("--out", default="data/rft_mix.jsonl")
ap.add_argument("--seed", type=int, default=3)
a = ap.parse_args()

rng = random.Random(a.seed)
rows = [json.loads(l) for l in open(a.rft)]
anchor = [json.loads(l) for l in open(a.anchor)]
rng.shuffle(anchor)
rows += anchor[: a.n_anchor]
rng.shuffle(rows)
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} -> {a.out}")
