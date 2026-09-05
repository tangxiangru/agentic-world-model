#!/usr/bin/env python3
"""Turn gen_vllm.py samples into rejection-sampling SFT rows.

Keeps only self-generated solutions whose last number (grader's own rule) equals
the gold answer, caps the number kept per question, and writes rows in the same
schema prep_data.py uses so train_sft.py can read either.
"""
from __future__ import annotations

import argparse
import json
import random
import re

from render import END, MATH_PROMPT_TEMPLATE

ANSWER_LINE = re.compile(r"(?m)^\s*ANSWER:\s*\$?-?[\d,]+(?:\.\d+)?\s*$")


def normalise(text: str) -> str:
    """Cheap dedup key: the sequence of numbers and operators in the solution."""
    return " ".join(re.findall(r"-?\d+\.?\d*|[-+*/=]", text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--max-chars", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows, stats = [], {"questions": 0, "with_any_correct": 0, "kept": 0, "dropped_format": 0}
    for line in open(args.samples):
        r = json.loads(line)
        stats["questions"] += 1
        cands = []
        seen = set()
        for comp, ok in zip(r["completions"], r["correct"]):
            if not ok:
                continue
            c = comp.strip()
            if c.endswith(END):
                c = c[: -len(END)].strip()
            if not (args.min_chars <= len(c) <= args.max_chars):
                stats["dropped_format"] += 1
                continue
            if c.count("ANSWER:") != 1 or not ANSWER_LINE.search(c):
                stats["dropped_format"] += 1
                continue
            key = normalise(c)
            if key in seen:
                continue
            seen.add(key)
            cands.append(c)
        if not cands:
            continue
        stats["with_any_correct"] += 1
        rng.shuffle(cands)
        # prefer the shortest surviving solutions: fewer tokens for the same reward
        cands.sort(key=len)
        for c in cands[: args.max_per_question]:
            rows.append(
                {
                    "prompt": MATH_PROMPT_TEMPLATE.format(prompt=r["question"].strip()),
                    "completion": c + END,
                    "answer": r["answer"],
                    "source": "synthetic:self",
                    "problem": r["question"],
                }
            )
    rng.shuffle(rows)
    stats["kept"] = len(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
