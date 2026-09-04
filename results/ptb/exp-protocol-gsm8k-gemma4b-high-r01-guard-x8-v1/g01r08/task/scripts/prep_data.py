#!/usr/bin/env python3
"""Build the SFT corpus for GSM8K.

Sources (all GSM8K *train* derived or independent; the GSM8K test split is
never read here):
  - nvidia/OpenMathInstruct-2, problem_source in {gsm8k, augmented_gsm8k}
  - openai/gsm8k train split minus the held-out dev300 tail

Every row is rendered into the exact prompt the grader uses
(inspect_evals.gsm8k.MATH_PROMPT_TEMPLATE) and a target that ends with
"ANSWER: <number>" so that the last number of the completion is the answer
(inspect_ai match(location="end", numeric=True)).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

from datasets import load_dataset

from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE

NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
BOXED_TAIL_RE = re.compile(
    r"\s*(?:so,?\s+)?(?:thus,?\s+)?(?:the\s+)?(?:final\s+)?answer is[^\n]*$", re.IGNORECASE | re.MULTILINE
)
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if NUM_RE.match(a):
        if a.endswith(".0"):
            a = a[:-2]
        return a
    return None


def clean_solution(sol: str) -> str:
    """Strip the dataset's own answer markers so ours is the only one."""
    sol = sol.strip()
    # unwrap \boxed{X} -> X (the grader never sees latex; keep the number)
    prev = None
    while prev != sol:
        prev = sol
        sol = BOXED_RE.sub(r"\1", sol)
    # drop a trailing "The final answer is ..." sentence
    sol = BOXED_TAIL_RE.sub("", sol).strip()
    return sol


def make_row(question: str, body: str, answer: str) -> dict | None:
    body = body.strip()
    if not body:
        return None
    if "\\boxed" in body or "ANSWER:" in body or "####" in body or "<end_of_turn>" in body:
        return None
    prompt = MATH_PROMPT_TEMPLATE.format(prompt=question.strip())
    target = f"{body}\n\nANSWER: {answer}<end_of_turn>"
    return {"prompt": prompt, "target": target, "question": question.strip(),
            "answer": answer}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_train.jsonl")
    ap.add_argument("--n-omi", type=int, default=120000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ---- held-out dev questions (never trained on) -------------------------
    dev_q = set()
    with open("data/dev_train300.jsonl") as f:
        for line in f:
            dev_q.add(json.loads(line)["question"].strip())

    rows: list[dict] = []

    # ---- 1. GSM8K train gold solutions ------------------------------------
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    n_gold = 0
    for r in gsm:
        q = r["question"].strip()
        if q in dev_q:
            continue
        body, _, ans = r["answer"].partition("####")
        ans = norm_answer(ans)
        if ans is None:
            continue
        body = re.sub(r"<<[^>]*>>", "", body).strip()
        row = make_row(q, body, ans)
        if row is not None:
            row["src"] = "gsm8k_train_gold"
            rows.append(row)
            n_gold += 1
    print(f"gsm8k gold: {n_gold}")

    # ---- 2. OpenMathInstruct-2, gsm8k-derived ------------------------------
    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    keep_sources = {"gsm8k", "augmented_gsm8k"}
    src_col = omi["problem_source"]
    sub_idx = [i for i, s in enumerate(src_col) if s in keep_sources]
    print("omi gsm8k-derived rows:", len(sub_idx))
    omi = omi.select(sub_idx).to_list()
    per_problem: dict[str, int] = defaultdict(int)
    idx = list(range(len(omi)))
    rng.shuffle(idx)
    n_omi = 0
    n_reject = defaultdict(int)
    for i in idx:
        if n_omi >= args.n_omi:
            break
        r = omi[i]
        q = r["problem"].strip()
        if q in dev_q:
            n_reject["dev"] += 1
            continue
        if per_problem[q] >= args.max_per_problem:
            n_reject["dup"] += 1
            continue
        ans = norm_answer(str(r["expected_answer"]))
        if ans is None:
            n_reject["answer"] += 1
            continue
        body = clean_solution(r["generated_solution"])
        if len(body) > args.max_chars:
            n_reject["long"] += 1
            continue
        row = make_row(q, body, ans)
        if row is None:
            n_reject["make_row"] += 1
            continue
        row["src"] = "omi2_gsm8k"
        per_problem[q] += 1
        rows.append(row)
        n_omi += 1
    print(f"omi2: {n_omi}, problems seen: {len(per_problem)}, rejects: {dict(n_reject)}")

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
