#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt on GSM8K.

Source: nvidia/OpenMathInstruct-2, restricted to the two gsm8k-derived
problem_source values. Those problems are seeded from the gsm8k TRAIN split
only; every produced row still goes through ../contamination_check.py before
training.

Every target is shaped exactly like the grader wants to read it: free-form
reasoning, then a final line "ANSWER: <number>". The prompt is the grader's own
MATH_PROMPT_TEMPLATE, and a configurable share of rows additionally carry the
grader's exact 10-shot system prefix so the model sees the graded prompt shape
during training as well.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq

SHARDS = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
GSM_SOURCES = {"gsm8k", "augmented_gsm8k"}

MATH_PROMPT_TEMPLATE = open("data/math_prompt_template.txt").read()
FEWSHOT_PREFIX = open("data/fewshot_system_message.txt").read()

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
INT_ANSWER = re.compile(r"^-?\d+(\.\d+)?$")


def clean_solution(sol: str, answer: str) -> str | None:
    """Turn a \\boxed{}-terminated OpenMathInstruct solution into a
    'reasoning ... \\n\\nANSWER: n' target with exactly one answer marker."""
    if sol.count("\\boxed") != 1:
        return None
    # drop the \boxed wrapper, keeping the sentence it sits in readable
    sol = BOXED.sub(r"\1", sol).strip()
    if "\\boxed" in sol or "ANSWER:" in sol:
        return None
    # the grader reads the LAST number in the completion; make it the answer.
    # The terminator is part of the target so that what is written to disk is
    # exactly what the trainer learns (see train_sft.render).
    return f"{sol}\n\nANSWER: {answer}<end_of_turn>"


def build_prompt(question: str, with_fewshot: bool) -> str:
    body = MATH_PROMPT_TEMPLATE.format(prompt=question).strip()
    if with_fewshot:
        return f"{FEWSHOT_PREFIX}\n\n{body}"
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_gsm.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--target", type=int, default=200000)
    ap.add_argument("--fewshot-share", type=float, default=0.10)
    ap.add_argument("--max-sol-chars", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    per_problem: dict[str, list[str]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    n_scanned = 0
    n_kept = 0

    for path in sorted(glob.glob(SHARDS)):
        tbl = pq.read_table(
            path, columns=["problem", "generated_solution", "expected_answer", "problem_source"]
        )
        df = tbl.to_pandas()
        df = df[df.problem_source.isin(GSM_SOURCES)]
        n_scanned += len(df)
        for problem, sol, ans in zip(df.problem, df.generated_solution, df.expected_answer):
            ans = (ans or "").strip()
            if not INT_ANSWER.match(ans):
                continue
            if len(sol) > args.max_sol_chars or len(problem) > 1500:
                continue
            if len(per_problem[problem]) >= args.max_per_problem:
                continue
            target = clean_solution(sol, ans)
            if target is None:
                continue
            key = (problem, target)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            per_problem[problem].append(target)
            n_kept += 1
        print(f"{path.split('/')[-1]}: scanned={n_scanned} kept={n_kept}", flush=True)
        if n_kept >= args.target * 1.15:
            break

    rows = [(p, t) for p, ts in per_problem.items() for t in ts]
    rng.shuffle(rows)
    rows = rows[: args.target]

    n_fewshot = int(len(rows) * args.fewshot_share)
    fewshot_idx = set(rng.sample(range(len(rows)), n_fewshot))
    with open(args.out, "w") as f:
        for i, (problem, target) in enumerate(rows):
            with_fewshot = i in fewshot_idx
            f.write(
                json.dumps(
                    {
                        "prompt": build_prompt(problem, with_fewshot),
                        "completion": target,
                        "fewshot": with_fewshot,
                    }
                )
                + "\n"
            )
    print(f"wrote {len(rows)} rows to {args.out} ({n_fewshot} with the 10-shot prefix)")
    print(f"unique problems: {len(per_problem)}")


if __name__ == "__main__":
    main()
