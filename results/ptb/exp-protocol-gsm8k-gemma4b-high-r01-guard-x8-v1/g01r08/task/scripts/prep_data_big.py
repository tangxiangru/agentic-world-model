#!/usr/bin/env python3
"""Larger GSM8K-style SFT corpus: the full OpenMathInstruct-2 train split,
gsm8k-derived problems only, one solution per problem to maximise unique
problems. Same formatting contract as prep_data.py.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from datasets import load_dataset

from prep_data import clean_solution, make_row, norm_answer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_train_big.jsonl")
    ap.add_argument("--n-omi", type=int, default=260000)
    ap.add_argument("--max-per-problem", type=int, default=1)
    ap.add_argument("--max-chars", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exclude", default=None, help="jsonl whose questions must not appear")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    dev_q = {json.loads(l)["question"].strip() for l in open("data/dev_train300.jsonl")}
    if args.exclude:
        for l in open(args.exclude):
            dev_q.add(json.loads(l)["question"].strip())
        print("exclude set:", len(dev_q), flush=True)

    rows = []
    # gsm8k train gold solutions
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    import re

    for r in (gsm if not args.exclude else []):
        q = r["question"].strip()
        if q in dev_q:
            continue
        body, _, ans = r["answer"].partition("####")
        ans = norm_answer(ans)
        if ans is None:
            continue
        row = make_row(q, re.sub(r"<<[^>]*>>", "", body).strip(), ans)
        if row:
            row["src"] = "gsm8k_train_gold"
            rows.append(row)
    print("gold:", len(rows), flush=True)

    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train")
    print("full split:", len(omi), flush=True)
    src = omi["problem_source"]
    idx = [i for i, s in enumerate(src) if s in ("gsm8k", "augmented_gsm8k")]
    print("gsm8k-derived:", len(idx), flush=True)
    rng.shuffle(idx)
    sub = omi.select(idx)
    per_problem: dict[str, int] = defaultdict(int)
    n = 0
    for r in sub:
        if n >= args.n_omi:
            break
        q = r["problem"].strip()
        if q in dev_q or per_problem[q] >= args.max_per_problem:
            continue
        ans = norm_answer(str(r["expected_answer"]))
        if ans is None:
            continue
        body = clean_solution(r["generated_solution"])
        if len(body) > args.max_chars:
            continue
        row = make_row(q, body, ans)
        if row is None:
            continue
        row["src"] = "omi2_gsm8k"
        per_problem[q] += 1
        rows.append(row)
        n += 1
        if n % 50000 == 0:
            print("built", n, flush=True)
    print("omi:", n, "unique problems:", len(per_problem), flush=True)

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(rows), "->", args.out, flush=True)

    with open(args.out.replace(".jsonl", "_for_decon.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")


if __name__ == "__main__":
    main()
