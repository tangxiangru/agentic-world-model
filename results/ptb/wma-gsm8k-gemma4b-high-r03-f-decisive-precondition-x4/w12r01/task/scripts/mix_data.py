#!/usr/bin/env python3
"""Mix jsonl row files with per-file caps into one training file."""
import argparse, json, random, sys
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--input", action="append", required=True,
                help="path[:max_rows]; max_rows 0 or omitted = all")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rng = random.Random(a.seed)
rows = []
for spec in a.input:
    path, _, cap = spec.partition(":")
    rs = [json.loads(l) for l in open(path)]
    rng.shuffle(rs)
    if cap and int(cap) > 0:
        rs = rs[: int(cap)]
    print(f"{path}: {len(rs)}", file=sys.stderr)
    rows.extend(rs)
rng.shuffle(rows)
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} -> {a.out}", file=sys.stderr)
print(Counter(r.get("source") for r in rows), file=sys.stderr)
