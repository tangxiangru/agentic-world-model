#!/usr/bin/env python3
"""Build SFT data for GSM8K from OpenMathInstruct-2 (and optional extra sources).

Targets are shaped exactly the way the grader reads them:
  * the assistant turn ends with '<end_of_turn>' (token 106, in eos_token_id)
  * the last line is 'ANSWER: <integer>' so match(numeric=True, location='end')
    reads the intended number.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re

import pandas as pd

OMI2 = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
ORCA = "/home/ben/hf_cache/hub/datasets--microsoft--orca-math-word-problems-200k/snapshots/*/data/*.parquet"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
INT_ANS = re.compile(r"^-?\d{1,12}$")


def clean_solution(sol: str, ans: str) -> str | None:
    s = sol.strip()
    # drop heavy-LaTeX solutions: grade-school style is plain prose + arithmetic
    if "\\[" in s or "\\begin{" in s or s.count("$") > 4 or "\\frac" in s:
        return None
    s = BOXED.sub(r"\1", s)
    if "\\boxed" in s:
        return None
    s = s.replace("$", "")
    s = re.sub(r"[ \t]+\n", "\n", s).strip()
    if not s:
        return None
    return s + "\n\nANSWER: " + ans


def load_omi2(max_per_source: dict[str, int], seed: int, sols_per_problem: int = 1) -> list[dict]:
    files = sorted(glob.glob(OMI2))
    assert files, "OpenMathInstruct-2 shards not found"
    keep_sources = set(max_per_source)
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=["problem", "generated_solution", "expected_answer", "problem_source"])
        df = df[df["problem_source"].isin(keep_sources)]
        df = df[df["expected_answer"].astype(str).str.match(INT_ANS)]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    rng = random.Random(seed)
    out: list[dict] = []
    for src, cap in max_per_source.items():
        sub = df[df["problem_source"] == src]
        # one solution per distinct problem, shuffled
        idx = list(range(len(sub)))
        rng.shuffle(idx)
        seen: dict[str, int] = {}
        rows = []
        for i in idx:
            r = sub.iloc[i]
            p = r["problem"].strip()
            if seen.get(p, 0) >= sols_per_problem:
                continue
            body = clean_solution(r["generated_solution"], str(r["expected_answer"]))
            if body is None:
                continue
            if len(body) > 2500 or len(p) > 1500:
                continue
            body_key = (p, body[:80])
            seen[p] = seen.get(p, 0) + 1
            rows.append({"question": p, "target": body, "source": src})
            if len(rows) >= cap:
                break
        print(f"  {src}: {len(rows)} rows (pool {len(sub)})")
        out.extend(rows)
    return out


def load_orca(cap: int, seed: int) -> list[dict]:
    files = sorted(glob.glob(ORCA))
    if not files:
        return []
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rng = random.Random(seed)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    rows = []
    for i in idx:
        r = df.iloc[i]
        q, a = str(r["question"]).strip(), str(r["answer"]).strip()
        # orca answers are free-form; keep only those whose final number is unambiguous
        nums = re.findall(r"-?\d[\d,]*\.?\d*", a.replace("**", ""))
        if not nums:
            continue
        final = nums[-1].replace(",", "")
        if not INT_ANS.match(final):
            continue
        body = clean_solution(a.replace("**", ""), final)
        if body is None or len(body) > 2500 or len(q) > 1500:
            continue
        rows.append({"question": q, "target": body, "source": "orca-math"})
        if len(rows) >= cap:
            break
    print(f"  orca-math: {len(rows)} rows")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm8k", type=int, default=15000)
    ap.add_argument("--n-aug-gsm8k", type=int, default=45000)
    ap.add_argument("--n-math", type=int, default=0)
    ap.add_argument("--n-aug-math", type=int, default=0)
    ap.add_argument("--n-orca", type=int, default=0)
    ap.add_argument("--fewshot-frac", type=float, default=0.10,
                    help="fraction of rows carrying the grader's exact 10-shot system prefix")
    ap.add_argument("--fewshot-file", default="data/fewshot_system.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sols-per-problem", type=int, default=1)
    args = ap.parse_args()

    caps = {}
    if args.n_gsm8k:
        caps["gsm8k"] = args.n_gsm8k
    if args.n_aug_gsm8k:
        caps["augmented_gsm8k"] = args.n_aug_gsm8k
    if args.n_math:
        caps["math"] = args.n_math
    if args.n_aug_math:
        caps["augmented_math"] = args.n_aug_math

    print("loading OpenMathInstruct-2 ...")
    rows = load_omi2(caps, args.seed, args.sols_per_problem) if caps else []
    if args.n_orca:
        print("loading orca-math ...")
        rows += load_orca(args.n_orca, args.seed)

    fewshot = open(args.fewshot_file).read().strip()
    rng = random.Random(args.seed + 1)
    rng.shuffle(rows)
    n_fs = int(len(rows) * args.fewshot_frac)
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            prompt_user = MATH_PROMPT_TEMPLATE.format(prompt=r["question"])
            system = fewshot if i < n_fs else None
            f.write(json.dumps({
                "system": system,
                "user": prompt_user,
                "completion": r["target"] + "<end_of_turn>",
                "source": r["source"],
            }) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} ({n_fs} with the 10-shot system prefix)")


if __name__ == "__main__":
    main()
