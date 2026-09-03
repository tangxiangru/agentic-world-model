#!/usr/bin/env python3
"""Concatenate jsonl SFT files, drop exact (prompt, completion) duplicates, shuffle."""
import argparse, hashlib, json, random
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rng = random.Random(a.seed)
seen, out = set(), []
for p in a.inputs:
    for line in open(p):
        r = json.loads(line)
        h = hashlib.md5((r["prompt"] + "\x00" + r["completion"]).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
rng.shuffle(out)
with open(a.out, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(out)} rows to {a.out}")
print(Counter(r["src"] for r in out))
