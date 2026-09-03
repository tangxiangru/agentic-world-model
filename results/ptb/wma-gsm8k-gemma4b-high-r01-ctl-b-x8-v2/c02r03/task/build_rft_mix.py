#!/usr/bin/env python3
"""Mix the rejection-sampled rows with SFT rows exp-02 never saw.

exp-02 trained on `random.Random(0).shuffle(rows); rows[:105000]` of
data/sft_v2.jsonl, so rows[105000:] is a held-out slice of the same corpus.
Mixing some of it in keeps the second round from collapsing onto the narrower
distribution of the model's own samples.
"""
from __future__ import annotations

import argparse
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", default="data/rft_v1.jsonl")
    ap.add_argument("--sft", default="data/sft_v2.jsonl")
    ap.add_argument("--seen-by-parent", type=int, default=105000)
    ap.add_argument("--n-fresh", type=int, default=20000)
    ap.add_argument("--out", default="data/rft_mix_v1.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rft = [json.loads(l) for l in open(args.rft)]

    sft = [json.loads(l) for l in open(args.sft)]
    random.Random(0).shuffle(sft)  # same shuffle the trainer used for exp-02
    fresh = sft[args.seen_by_parent :]
    rng = random.Random(args.seed)
    rng.shuffle(fresh)
    fresh = fresh[: args.n_fresh]

    rows = rft + fresh
    rng.shuffle(rows)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n_fs = sum(1 for r in rows if r.get("system"))
    print(f"rft={len(rft)} fresh_sft={len(fresh)} total={len(rows)} fewshot_prefixed={n_fs} -> {args.out}")


if __name__ == "__main__":
    main()
