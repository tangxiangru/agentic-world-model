#!/usr/bin/env python3
"""Build the SFT set: OpenMathInstruct-2 (gsm8k-derived) reformatted to the grader's format.

Output jsonl rows: {"question", "completion", "answer", "source"}
`completion` is the assistant turn WITHOUT the stop token; the trainer appends it.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict

import pandas as pd
from huggingface_hub import hf_hub_download

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
NUMLIKE = re.compile(r"^-?\d{1,12}(\.\d+)?$")


def unbox(sol: str) -> str:
    # keep the inner text inline so the sentence still reads naturally
    return BOXED.sub(lambda m: m.group(1), sol)


def clean_answer(a: str) -> str | None:
    a = str(a).strip().replace(",", "").replace("$", "").replace("%", "")
    if not NUMLIKE.match(a):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    if "." in a:
        a = a.rstrip("0").rstrip(".")
    return a or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--per-problem", type=int, default=1)
    ap.add_argument("--n-gsm", type=int, default=45000)
    ap.add_argument("--n-math", type=int, default=5000)
    ap.add_argument("--max-sol-chars", type=int, default=2600)
    ap.add_argument("--out", default="data/sft_raw.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    frames = []
    for i in range(args.shards):
        p = hf_hub_download(
            "nvidia/OpenMathInstruct-2",
            f"data/train-{i:05d}-of-00032.parquet",
            repo_type="dataset",
        )
        frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    print("loaded", df.shape)

    gsm_mask = df.problem_source.isin(["gsm8k", "augmented_gsm8k"])
    math_mask = df.problem_source.isin(["math", "augmented_math"])

    def collect(sub: pd.DataFrame, budget: int, tag: str) -> list[dict]:
        by_problem: dict[str, list[dict]] = defaultdict(list)
        for prob, sol, ans in zip(
            sub.problem.values, sub.generated_solution.values, sub.expected_answer.values
        ):
            a = clean_answer(ans)
            if a is None:
                continue
            sol = str(sol)
            if len(sol) > args.max_sol_chars or len(sol) < 40:
                continue
            if "ANSWER:" in sol or "####" in sol:
                continue
            body = unbox(sol).strip()
            if "\\boxed" in body or "\\begin" in body or "$$" in body:
                continue
            # the grader reads the LAST number: guarantee it is the answer
            comp = body + "\nANSWER: " + a
            by_problem[str(prob)].append({"c": comp, "a": a})
        keys = list(by_problem.keys())
        rng.shuffle(keys)
        out = []
        for k in keys:
            if len(out) >= budget:
                break
            cands = by_problem[k]
            rng.shuffle(cands)
            # prefer mid-length solutions: not the terse ones, not the rambling ones
            cands.sort(key=lambda d: len(d["c"]))
            pick = cands[len(cands) // 2 : len(cands) // 2 + args.per_problem]
            for d in pick:
                out.append(
                    {"question": k, "completion": d["c"], "answer": d["a"], "source": tag}
                )
        return out

    rows = collect(df[gsm_mask], args.n_gsm, "omi2-gsm8k")
    print("gsm rows", len(rows))
    mrows = collect(df[math_mask], args.n_math, "omi2-math")
    print("math rows", len(mrows))
    rows += mrows
    rng.shuffle(rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(rows), "->", args.out)


if __name__ == "__main__":
    main()
