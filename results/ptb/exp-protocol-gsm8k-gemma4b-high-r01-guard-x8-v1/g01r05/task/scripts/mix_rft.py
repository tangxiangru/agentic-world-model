#!/usr/bin/env python3
"""Assemble the exp-03 training mixture: rejection-sampled self-solutions plus a
replay slice of OpenMathInstruct-2 solutions the exp-02 run never saw."""
from __future__ import annotations

import argparse
import json
import random

from build_sft_data import fewshot_pool


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", required=True)
    ap.add_argument("--seen", required=True, help="sft_v2.jsonl - rows already trained on")
    ap.add_argument("--pool", required=True, help="sft_v3.jsonl built with --max-per-problem 2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-replay", type=int, default=20000)
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    ap.add_argument("--fewshot-ks", type=str, default="1,2,4")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rft = load(args.rft)
    seen = {r["completion_body"] for r in load(args.seen)}
    fresh = [r for r in load(args.pool) if r["completion_body"] not in seen]
    rng.shuffle(fresh)
    replay = fresh[: args.n_replay]
    for r in replay:
        r["source"] = "replay:" + r.get("source", "omi2")

    rows = rft + replay
    rng.shuffle(rows)

    if args.fewshot_frac > 0:
        pool = fewshot_pool()
        ks = [int(k) for k in args.fewshot_ks.split(",")]
        n_fs = int(len(rows) * args.fewshot_frac)
        for r in rows[:n_fs]:
            shots = rng.sample(pool, rng.choice(ks))
            r["prompt"] = "\n\n".join(shots) + "\n\n" + r["prompt"]
        rng.shuffle(rows)
        print(f"prefixed {n_fs} rows with few-shot examples")

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"wrote {len(rows)} -> {args.out}")
    print(Counter(r["source"] for r in rows))


if __name__ == "__main__":
    main()
