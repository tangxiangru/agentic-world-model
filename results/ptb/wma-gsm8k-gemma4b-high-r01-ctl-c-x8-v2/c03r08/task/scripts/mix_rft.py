#!/usr/bin/env python3
"""Mix RFT rows with a random slice of the SFT corpus (anti-narrowing ballast)."""
import argparse, json, random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--rft", required=True)
ap.add_argument("--sft", required=True)
ap.add_argument("--n-sft", type=int, default=25000)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rng = random.Random(a.seed)
rft = [json.loads(l) for l in Path(a.rft).open()]
sft = [json.loads(l) for l in Path(a.sft).open()]
rng.shuffle(sft)
rows = rft + sft[: a.n_sft]
rng.shuffle(rows)
with Path(a.out).open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"rft {len(rft)} + sft {min(a.n_sft, len(sft))} = {len(rows)} -> {a.out}")
