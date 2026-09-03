#!/usr/bin/env python3
"""Mix jsonl training files: --in path:fraction (fraction of that file's rows)."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="ins", nargs="+", required=True, help="path:fraction")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rng = random.Random(a.seed)
rows = []
for spec in a.ins:
    path, _, frac = spec.rpartition(":")
    frac = float(frac)
    rs = [json.loads(l) for l in open(path)]
    rng.shuffle(rs)
    k = int(len(rs) * frac) if frac <= 1.0 else int(frac)
    rows += rs[:k]
    print(f"{path}: {len(rs)} -> {k}")

rng.shuffle(rows)
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"],
                            "source": r.get("source", "?")}) + "\n")
print(a.out, len(rows), Counter(r.get("source", "?") for r in rows))
