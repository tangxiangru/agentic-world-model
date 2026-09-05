#!/usr/bin/env python3
"""Turn rejection-sampled solutions into an SFT-format jsonl."""
from __future__ import annotations

import argparse
import json
import random

from prep_data import MATH_PROMPT_TEMPLATE, gsm8k_fewshot_blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for path in args.inp:
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    rng.shuffle(rows)
    fewshot_pool = gsm8k_fewshot_blocks(600, 0)

    counts = {}
    with open(args.out, "w") as f:
        for r in rows:
            counts[r["source"]] = counts.get(r["source"], 0) + 1
            system = ""
            if rng.random() < args.fewshot_frac:
                k = rng.randint(2, 10)
                system = "\n\n".join(rng.sample(fewshot_pool, k))
            f.write(
                json.dumps(
                    {
                        "system": system,
                        "user": MATH_PROMPT_TEMPLATE.format(prompt=r["problem"]),
                        "assistant": r["solution"],
                        "answer": r["answer"],
                        "source": r["source"],
                    }
                )
                + "\n"
            )
    print(f"wrote {len(rows)} -> {args.out}  {counts}")


if __name__ == "__main__":
    main()
