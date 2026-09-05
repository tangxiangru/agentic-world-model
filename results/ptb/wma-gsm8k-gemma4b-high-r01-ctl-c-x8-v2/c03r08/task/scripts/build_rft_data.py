#!/usr/bin/env python3
"""Turn gen_vllm.py sample files into an SFT corpus (rejection-sampling fine-tuning).

Keeps only samples whose graded answer is correct, caps how many survive per
problem, and re-attaches the stop token so the row is byte-identical in shape to
data/sft_v1.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from build_sft_data import END_OF_TURN, MATH_PROMPT_TEMPLATE, build_user

ANSWER_LINE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")


def clean(text: str) -> str | None:
    """Keep well-formed chains: exactly one ANSWER marker, and it ends the text."""
    t = text.strip()
    if t.count("ANSWER:") != 1:
        return None
    if not ANSWER_LINE.search(t):
        return None
    if len(t) < 40 or len(t) > 3500:
        return None
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-hard", action="store_true",
                    help="keep only problems the sampler did NOT solve every time")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kept, stats = [], Counter()
    for path in args.samples:
        for line in Path(path).open():
            r = json.loads(line)
            stats["problems"] += 1
            flags = r["correct"]
            if not any(flags):
                stats["unsolved"] += 1
                continue
            if args.only_hard and all(flags):
                stats["skipped_easy"] += 1
                continue
            good, seen = [], set()
            for txt, ok in zip(r["samples"], flags):
                if not ok:
                    continue
                c = clean(txt)
                if c is None or c in seen:
                    continue
                seen.add(c)
                good.append(c)
            if not good:
                stats["no_clean_sample"] += 1
                continue
            rng.shuffle(good)
            for c in good[: args.per_problem]:
                kept.append({"question": r["question"],
                             "prompt": build_user(r["question"]),
                             "completion": c + END_OF_TURN,
                             "gold": r["gold"],
                             "src": "rft_self"})
            stats["problems_kept"] += 1
    rng.shuffle(kept)
    with Path(args.out).open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(dict(stats), indent=1))
    print(f"wrote {len(kept)} rows -> {args.out}")


if __name__ == "__main__":
    main()
