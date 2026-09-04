#!/usr/bin/env python3
"""Build a corpus of GSM8K-train-derived rows that no earlier card has trained on.

Draws from OpenMathInstruct-2's train_5M split (806k gsm8k + augmented_gsm8k
rows against the 153k reachable in train_1M) and excludes, by problem text,
every problem already present in the corpora of exp-02 and exp-03. Same
filters, same rendering, same terminator as build_data.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

from datasets import load_dataset  # noqa: E402

from build_data import MATH_PROMPT_TEMPLATE, build_target, fewshot_prefix, norm_answer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v4.jsonl")
    ap.add_argument("--n-rows", type=int, default=110000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=["data/sft_v1.jsonl", "data/sft_v3.jsonl"])
    ap.add_argument("--carry", nargs="*", default=[], help="jsonl rows to append verbatim")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    prefix = fewshot_prefix()

    seen_problems: set[str] = set()
    for p in args.exclude:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for l in f:
                seen_problems.add(json.loads(l)["user"])
    print(f"excluding {len(seen_problems)} already-trained prompts")

    d = load_dataset("nvidia/OpenMathInstruct-2", split="train_5M")
    d = d.filter(lambda r: r["problem_source"] in ("gsm8k", "augmented_gsm8k"), num_proc=16)
    print(f"gsm8k-derived rows available: {len(d)}")
    idx = list(range(len(d)))
    rng.shuffle(idx)

    per_problem: dict[str, int] = {}
    out = []
    for i in idx:
        if len(out) >= args.n_rows:
            break
        r = d[i]
        ans = norm_answer(r["expected_answer"])
        if ans is None:
            continue
        q = r["problem"].strip()
        user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        if user in seen_problems:
            continue
        if per_problem.get(q, 0) >= args.max_per_problem:
            continue
        tgt = build_target(r["generated_solution"], ans)
        if tgt is None:
            continue
        per_problem[q] = per_problem.get(q, 0) + 1
        use_fs = rng.random() < args.fewshot_frac
        out.append(
            {
                "id": f"v4-{len(out)}",
                "system": prefix if use_fs else None,
                "user": user,
                "target": tgt,
                "answer": ans,
                "src": r["problem_source"],
                "fewshot": bool(use_fs),
            }
        )
    print(f"new teacher rows: {len(out)} over {len(per_problem)} distinct problems")

    for p in args.carry:
        with open(p) as f:
            add = [json.loads(l) for l in f]
        out.extend(add)
        print(f"carried {len(add)} rows from {p}")

    rng.shuffle(out)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    with open(args.out.replace(".jsonl", "_forcheck.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps({"text": r["user"] + "\n" + r["target"]}) + "\n")
    print(f"wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
