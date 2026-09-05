#!/usr/bin/env python3
"""Turn vllm_gen.py --mode sample output into an SFT file (rejection sampling).

Keeps only completions whose last number equals the gold answer and whose last
line is a well-formed ANSWER line, dedups near-identical solutions per problem,
and caps solutions per problem.
"""
from __future__ import annotations

import argparse
import json
import random
import re

from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    d = json.load(open(args.samples))
    rows, n_seen, n_kept = [], 0, 0
    for rec in d["records"]:
        good = []
        for c in rec["cands"]:
            n_seen += 1
            t = c["text"].strip()
            if not c["correct"] or not c["fmt_ok"]:
                continue
            if len(t) > args.max_chars or "ANSWER:" not in t:
                continue
            if t.count("ANSWER:") != 1:
                continue
            good.append(t)
        # dedup by the sequence of numbers used in the solution
        seen = set()
        uniq = []
        for t in good:
            key = tuple(re.findall(r"-?\d+(?:\.\d+)?", t))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(t)
        rng.shuffle(uniq)
        for t in uniq[: args.max_per_problem]:
            rows.append(
                {
                    "prompt": MATH_PROMPT_TEMPLATE.format(prompt=rec["question"].strip()),
                    "target": t + "<end_of_turn>",
                    "question": rec["question"].strip(),
                    "answer": rec["gold"],
                    "src": "rft_self",
                }
            )
            n_kept += 1
    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(args.out.replace(".jsonl", "_for_decon.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")
    print(f"candidates {n_seen}, kept {n_kept}, problems covered "
          f"{len({r['question'] for r in rows})} -> {args.out}")


if __name__ == "__main__":
    main()
