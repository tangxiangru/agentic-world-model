#!/usr/bin/env python3
"""Build SFT data in the exact inspect_evals/gsm8k output format.

Sources (all GSM8K *train*-derived or independent synthetic math; never test):
  - nvidia/OpenMathInstruct-2  (gsm8k, augmented_gsm8k, math, augmented_math)
  - openai/gsm8k train split (original human CoT)
"""
import argparse
import json
import random
import re

from datasets import load_dataset

from common import norm_num

BOXED = re.compile(r"\\boxed\{")


def strip_boxed(sol: str) -> str:
    """Remove \\boxed{...} wrappers, keeping inner content."""
    out = []
    i = 0
    while i < len(sol):
        m = BOXED.search(sol, i)
        if not m:
            out.append(sol[i:])
            break
        out.append(sol[i:m.start()])
        j = m.end()
        depth = 1
        inner = []
        while j < len(sol) and depth:
            if sol[j] == "{":
                depth += 1
            elif sol[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            inner.append(sol[j])
            j += 1
        out.append("".join(inner))
        i = j + 1
    return "".join(out)


LATEX_SUBS = [
    (re.compile(r"\\\[|\\\]"), ""),
    (re.compile(r"\\\(|\\\)"), ""),
    (re.compile(r"\\text\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\dfrac"), r"\\frac"),
    (re.compile(r"\\times"), "*"),
    (re.compile(r"\\cdot"), "*"),
    (re.compile(r"\\div"), "/"),
    (re.compile(r"\\%"), "%"),
    (re.compile(r"\\\$"), "$"),
    (re.compile(r"\\left|\\right"), ""),
    (re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}"), r"\1/\2"),
    (re.compile(r"[ \t]+\n"), "\n"),
    (re.compile(r"\n{3,}"), "\n\n"),
]


def clean_solution(sol: str) -> str:
    sol = strip_boxed(sol)
    for pat, rep in LATEX_SUBS:
        sol = pat.sub(rep, sol)
    # drop trailing "The answer is ..." style leftovers; we append our own line
    sol = sol.strip()
    return sol


def is_clean_numeric(ans: str) -> bool:
    a = ans.strip().replace(",", "").replace("$", "")
    try:
        f = float(a)
    except ValueError:
        return False
    return abs(f) < 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--n-gsm", type=int, default=14764)
    ap.add_argument("--n-aug-gsm", type=int, default=100000)
    ap.add_argument("--n-math", type=int, default=8000)
    ap.add_argument("--n-aug-math", type=int, default=20000)
    ap.add_argument("--max-chars", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")

    caps = {
        "gsm8k": args.n_gsm,
        "augmented_gsm8k": args.n_aug_gsm,
        "math": args.n_math,
        "augmented_math": args.n_aug_math,
    }
    buckets = {k: [] for k in caps}

    cols = ds.to_dict()
    n = len(cols["problem"])
    order = list(range(n))
    rng.shuffle(order)

    seen_pairs = set()
    for i in order:
        src = cols["problem_source"][i]
        if src not in buckets or len(buckets[src]) >= caps[src]:
            continue
        prob = cols["problem"][i]
        sol = cols["generated_solution"][i]
        ans = cols["expected_answer"][i]
        if not is_clean_numeric(ans):
            continue
        if "\\boxed" not in sol:
            continue
        if len(sol) > args.max_chars or len(prob) > 1500:
            continue
        # keep at most 2 solutions per problem for diversity without redundancy
        key = (prob, len(buckets[src]) and sol[:40])
        h = (prob,)
        cnt = seen_pairs
        body = clean_solution(sol)
        if not body:
            continue
        buckets[src].append({
            "question": prob.strip(),
            "solution": body,
            "answer": norm_num(ans),
            "source": src,
        })

    # limit duplicate problems: keep max 2 solutions per unique problem
    out = []
    per_problem = {}
    for src, items in buckets.items():
        for it in items:
            k = it["question"]
            per_problem.setdefault(k, 0)
            if per_problem[k] >= 2:
                continue
            per_problem[k] += 1
            out.append(it)

    # original GSM8K train CoT (rewritten to remove calculator annotations)
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for rec in gsm:
        q = rec["question"].strip()
        a = rec["answer"]
        body, tgt = a.split("####")
        body = re.sub(r"<<[^>]*>>", "", body).strip()
        out.append({
            "question": q,
            "solution": body,
            "answer": norm_num(tgt),
            "source": "gsm8k_orig",
        })

    rng.shuffle(out)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for it in out:
            f.write(json.dumps(it) + "\n")

    from collections import Counter
    print(Counter(x["source"] for x in out))
    print("total", len(out))


if __name__ == "__main__":
    main()
