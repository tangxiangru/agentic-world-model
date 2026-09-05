#!/usr/bin/env python3
"""Mix rejection-sampled (on-policy) data with a replay slice of the original
SFT pool for the second training round."""
import argparse, json, random
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--rft", default="work/rft_data.jsonl")
ap.add_argument("--base", default="work/sft_data.jsonl")
ap.add_argument("--out", default="work/v2_data.jsonl")
ap.add_argument("--n-replay", type=int, default=45000)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

random.seed(args.seed)
rft = [json.loads(l) for l in open(args.rft)]
base = [json.loads(l) for l in open(args.base)]

# Prefer replay from gsm8k-derived sources; keep some MATH for generality.
gsm_base = [r for r in base if r["src"] in ("gsm8k", "augmented_gsm8k", "gsm8k_orig")]
math_base = [r for r in base if r["src"] in ("math", "augmented_math")]
random.shuffle(gsm_base)
random.shuffle(math_base)
n_math = min(len(math_base), args.n_replay // 5)
replay = gsm_base[: args.n_replay - n_math] + math_base[:n_math]

out = rft + replay
random.shuffle(out)
with open(args.out, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(Counter(r["src"] for r in out))
print("total", len(out), "->", args.out)
