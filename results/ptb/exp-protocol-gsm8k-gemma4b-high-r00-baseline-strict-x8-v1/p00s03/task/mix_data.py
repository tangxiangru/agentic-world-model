#!/usr/bin/env python3
"""Concatenate jsonl training files with per-file row caps and shuffle."""
from __future__ import annotations

import argparse
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True, help="path:cap (cap 0 = all)")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rng = random.Random(a.seed)
rows = []
for spec in a.inputs:
    path, _, cap = spec.rpartition(":")
    cap = int(cap)
    rs = [json.loads(l) for l in open(path)]
    rng.shuffle(rs)
    if cap:
        rs = rs[:cap]
    print(f"{path}: {len(rs)} rows")
    rows += rs
rng.shuffle(rows)
with open(a.out, "w") as fo, open(a.out.replace(".jsonl", ".decon.jsonl"), "w") as fd:
    for r in rows:
        fo.write(json.dumps(r) + "\n")
        fd.write(json.dumps({"question": r["question"], "answer": r["target"]}) + "\n")
print("wrote", a.out, len(rows))
