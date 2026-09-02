#!/usr/bin/env python3
"""Interleave several prompt/completion jsonl files with given row counts."""
import argparse, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True, help="file[:n_rows] ...")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rng = random.Random(a.seed)
rows = []
for spec in a.inputs:
    path, _, n = spec.partition(":")
    part = [json.loads(l) for l in open(path)]
    rng.shuffle(part)
    if n:
        part = part[: int(n)]
    for r in part:
        r["mix_src"] = path
    rows += part
    print(f"{path}: {len(part)} rows")
rng.shuffle(rows)
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} -> {a.out}")
