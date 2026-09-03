#!/usr/bin/env python3
"""Second teacher corpus: gsm8k-derived OpenMathInstruct-2 problems that appear
in NO earlier corpus of this batch.

Same formatting rules as prepare_data.py (one 'ANSWER: n' line, terminated by
<end_of_turn>, grader's MATH_PROMPT_TEMPLATE, a small share carrying the
grader's verbatim 10-shot prefix). The only difference is the exclusion set:
every problem already seen by exp-02 or exp-04, plus the dev300 watch set.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq

GSM_SOURCES = {"gsm8k", "augmented_gsm8k"}
SPLIT = (
    'Solve the following math problem step by step. The last line of your response '
    'should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the '
    "answer to the problem.\n\n"
)
MATH_PROMPT_TEMPLATE = open("data/math_prompt_template.txt").read()
FEWSHOT_PREFIX = open("data/fewshot_system_message.txt").read()
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
INT_ANSWER = re.compile(r"^-?\d+(\.\d+)?$")


def clean_solution(sol: str, answer: str) -> str | None:
    if sol.count("\\boxed") != 1:
        return None
    sol = BOXED.sub(r"\1", sol).strip()
    if "\\boxed" in sol or "ANSWER:" in sol:
        return None
    return f"{sol}\n\nANSWER: {answer}<end_of_turn>"


def question_of(prompt: str) -> str | None:
    if SPLIT not in prompt:
        return None
    return prompt.split(SPLIT, 1)[1].split("\n\nRemember to put your answer")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_exp05.jsonl")
    ap.add_argument("--target", type=int, default=60000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-share", type=float, default=0.06)
    ap.add_argument("--max-sol-chars", type=int, default=2000)
    ap.add_argument("--first-shard", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seen: set[str] = set()
    for line in open("data/sft_gsm_clean.jsonl"):
        q = question_of(json.loads(line)["prompt"])
        if q:
            seen.add(q)
    for line in open("data/dev300.jsonl"):
        seen.add(json.loads(line)["question"])
    print(f"exclusion set: {len(seen)} problems already used or held out")

    shards = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
    ))
    shards = [s for s in shards if int(s.split("train-")[1][:5]) >= args.first_shard]

    rng = random.Random(args.seed)
    per_problem: dict[str, list[str]] = defaultdict(list)
    n_kept = 0
    for path in shards:
        df = pq.read_table(
            path, columns=["problem", "generated_solution", "expected_answer", "problem_source"]
        ).to_pandas()
        df = df[df.problem_source.isin(GSM_SOURCES)]
        for problem, sol, ans in zip(df.problem, df.generated_solution, df.expected_answer):
            if problem in seen:
                continue
            ans = (ans or "").strip()
            if not INT_ANSWER.match(ans):
                continue
            if len(sol) > args.max_sol_chars or len(problem) > 1500:
                continue
            if len(per_problem[problem]) >= args.max_per_problem:
                continue
            target = clean_solution(sol, ans)
            if target is None or target in per_problem[problem]:
                continue
            per_problem[problem].append(target)
            n_kept += 1
        print(f"{path.split('/')[-1]}: kept={n_kept}", flush=True)
        if n_kept >= args.target * 1.1:
            break

    rows = [(p, t) for p, ts in per_problem.items() for t in ts]
    rng.shuffle(rows)
    rows = rows[: args.target]
    n_few = int(len(rows) * args.fewshot_share)
    few_idx = set(rng.sample(range(len(rows)), n_few))
    with open(args.out, "w") as f:
        for i, (problem, target) in enumerate(rows):
            fs = i in few_idx
            body = MATH_PROMPT_TEMPLATE.format(prompt=problem).strip()
            prompt = f"{FEWSHOT_PREFIX}\n\n{body}" if fs else body
            f.write(json.dumps({"prompt": prompt, "completion": target, "fewshot": fs}) + "\n")
    print(f"wrote {len(rows)} rows ({n_few} with the 10-shot prefix) over "
          f"{len(per_problem)} fresh problems to {args.out}")


if __name__ == "__main__":
    main()
