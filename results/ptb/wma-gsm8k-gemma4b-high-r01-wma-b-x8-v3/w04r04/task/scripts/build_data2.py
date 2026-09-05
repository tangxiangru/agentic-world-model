#!/usr/bin/env python3
"""Round-2 mixture: self-generated rejection-sampled rows + *fresh* OpenMathInstruct-2
gsm8k-derived problems that round 1 never saw.

Reads the round-1 jsonl only to learn which problems are already spent. Never reads the
GSM8K test split.
"""
from __future__ import annotations

import argparse
import json
import random
import sys

sys.path.insert(0, "/home/ben/task/scripts")
from build_data import MATH_PROMPT_TEMPLATE, make_row, unwrap_boxed, build_gsm8k_train, sample_to_fewshot  # noqa: E402

from datasets import load_from_disk  # noqa: E402


def question_of(prompt: str) -> str:
    pre, post = MATH_PROMPT_TEMPLATE.split("{prompt}")
    return prompt[len(pre):len(prompt) - len(post)].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", default=None, help="jsonl from scripts/sample_rft.py")
    ap.add_argument("--seen", action="append", default=[], help="round-1 jsonl(s) to exclude")
    ap.add_argument("--omi2-dir", default="/home/ben/task/data/omi2_gsm_full")
    ap.add_argument("--n-fresh", type=int, default=60000)
    ap.add_argument("--per-problem", type=int, default=1)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # OpenMathInstruct-2's gsm8k portion covers only ~80k distinct problems with ~30
    # solutions each, and round 1 already used almost every problem. "Fresh" therefore
    # means a (problem, solution) pair round 1 did not use, not a new problem.
    seen: set[tuple[str, str]] = set()
    seen_q: set[str] = set()
    for p in args.seen:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                q = question_of(r["prompt"])
                seen_q.add(q)
                seen.add((q, r["completion"]))
    print(f"seen problems: {len(seen_q)}, seen (problem, solution) pairs: {len(seen)}")

    rows = []
    if args.rft:
        with open(args.rft) as f:
            for line in f:
                rows.append(json.loads(line))
    n_rft = len(rows)
    print(f"rft rows: {n_rft}")

    fresh, per = [], {}
    d = load_from_disk(args.omi2_dir)
    order = list(range(len(d)))
    rng.shuffle(order)
    for i in order:
        if len(fresh) >= args.n_fresh:
            break
        r = d[i]
        q = r["problem"].strip()
        if per.get(q, 0) >= args.per_problem:
            continue
        sol = unwrap_boxed(r["generated_solution"])
        if sol is None or "\\boxed" in sol or len(sol) > 2600:
            continue
        row = make_row(q, sol, r["expected_answer"], r["problem_source"])
        if row is None or (q, row["completion"]) in seen:
            continue
        per[q] = per.get(q, 0) + 1
        fresh.append({
            "system": None,
            "prompt": MATH_PROMPT_TEMPLATE.format(prompt=row["question"]),
            "completion": row["completion"],
            "answer": row["answer"],
            "src": "fresh:" + row["src"],
        })
    print(f"fresh rows: {len(fresh)}")

    rows += fresh
    rng.shuffle(rows)

    gsm = build_gsm8k_train()
    pool = [sample_to_fewshot(r["question"], r["completion"].rsplit("\n\nANSWER:", 1)[0], r["answer"])
            for r in gsm]

    n_fs = 0
    with open(args.out, "w") as f:
        for r in rows:
            if rng.random() < args.fewshot_frac:
                k = rng.choice([1, 2, 3])
                r["system"] = "\n\n".join(rng.sample(pool, k))
                n_fs += 1
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} ({n_rft} rft + {len(fresh)} fresh, "
          f"{n_fs} with a few-shot prefix)")


if __name__ == "__main__":
    main()
