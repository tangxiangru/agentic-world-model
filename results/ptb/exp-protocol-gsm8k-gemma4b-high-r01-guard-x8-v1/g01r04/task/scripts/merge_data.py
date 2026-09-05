#!/usr/bin/env python3
"""Merge SFT files into one training file, re-assign the few-shot-prefixed
slice, and write the contamination-checker input alongside it."""
from __future__ import annotations

import argparse
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n-fewshot", type=int, default=4000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rows = []
for p in args.inputs:
    n = 0
    for line in open(p):
        r = json.loads(line)
        rows.append({"question": r["question"], "completion": r["completion"],
                     "src": r.get("src", p)})
        n += 1
    print(f"{p}: {n}")

rng = random.Random(args.seed)
rng.shuffle(rows)
for i, r in enumerate(rows):
    r["system_mode"] = "fewshot" if i < args.n_fewshot else "zeroshot"
rng.shuffle(rows)

with open(args.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
with open(args.out.replace(".jsonl", ".check.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps({"text": r["question"] + "\n" + r["completion"]}) + "\n")

import collections  # noqa: E402
print(collections.Counter(r["src"] for r in rows))
print(f"wrote {len(rows)} -> {args.out}")
