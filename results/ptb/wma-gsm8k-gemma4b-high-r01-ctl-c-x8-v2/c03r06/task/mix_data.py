#!/usr/bin/env python3
"""Mix jsonl SFT shards, re-render a fraction with a few-shot system prefix,
shuffle, and write one training file."""
from __future__ import annotations

import argparse
import json
import random

from build_data import (MATH_PROMPT_TEMPLATE, fewshot_block, gsm8k_train_rows,
                        render_prompt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", action="append", default=[],
                    help="path[:n] - take n rows (all if omitted)")
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for spec in args.shard:
        path, _, n = spec.partition(":")
        got = [json.loads(l) for l in open(path)]
        rng.shuffle(got)
        if n:
            got = got[: int(n)]
        print(f"[mix] {path}: {len(got)} rows")
        rows.extend(got)

    shot_pool = list(gsm8k_train_rows())[:2000]

    n_fs = 0
    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            if rng.random() < args.fewshot_frac and "question" in r:
                k = rng.choice([2, 3, 4, 5, 8, 10])
                shots = rng.sample(shot_pool, k)
                system = "\n\n".join(
                    fewshot_block(s["q"], s["sol"], s["ans"]) for s in shots)
                r = dict(r, prompt=render_prompt(
                    system, MATH_PROMPT_TEMPLATE.format(prompt=r["question"])))
                n_fs += 1
            f.write(json.dumps(r) + "\n")
    print(f"[mix] wrote {len(rows)} rows ({n_fs} with a few-shot prefix) to {args.out}")


if __name__ == "__main__":
    main()
