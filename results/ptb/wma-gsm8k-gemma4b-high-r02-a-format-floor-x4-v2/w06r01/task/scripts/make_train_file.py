#!/usr/bin/env python3
"""Freeze the exact rows a training card will read.

Targets are written with the terminator already attached, so the file itself
records the stop token the preflight check verifies.
"""
import argparse, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="/home/ben/task/data/pool_clean.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-per-problem", type=int, default=99)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.pool)]
random.Random(a.seed).shuffle(rows)
seen, out = {}, []
for r in rows:
    q = r["question"]
    if seen.get(q, 0) >= a.max_per_problem:
        continue
    seen[q] = seen.get(q, 0) + 1
    r["target"] = r["target"].strip() + "<end_of_turn>"
    out.append(r)
    if len(out) >= a.n:
        break
with open(a.out, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(out)} rows to {a.out} ({len(seen)} unique problems)")
