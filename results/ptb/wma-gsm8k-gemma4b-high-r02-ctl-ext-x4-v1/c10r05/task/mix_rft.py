#!/usr/bin/env python3
"""Mix the RFT rows with a slice of unseen OpenMathInstruct-2 rows and shuffle."""
import argparse
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--rft", required=True)
ap.add_argument("--fresh", required=True)
ap.add_argument("--n-fresh", type=int, default=40000)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rng = random.Random(a.seed)
rows = [json.loads(l) for l in open(a.rft)]
n_rft = len(rows)
fresh = [json.loads(l) for l in open(a.fresh)]
rng.shuffle(fresh)
rows += fresh[: a.n_fresh]
seen, out = set(), []
for r in rows:
    if r["id"] in seen:
        continue
    seen.add(r["id"])
    out.append(r)
rng.shuffle(out)
with open(a.out, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"rft={n_rft} fresh={min(a.n_fresh, len(fresh))} total_after_dedup={len(out)} -> {a.out}")
