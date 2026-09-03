#!/usr/bin/env python3
"""exp-05 corpus: teacher supervision aimed at a *measured* weak spot.

Shards 14-27 turned out to contain only 135 gsm8k-derived problems that shards
0-13 did not - the dataset has ~80k distinct gsm8k-family problems and many
solutions each, so "more teacher problems" is not on the table. What is on the
table is which problems the teacher solutions are spent on, and how many
solutions each gets.

Three components, all with OpenMathInstruct-2's own verified expected_answer:
  hard      up to 6 distinct solutions for each of the 4,385 problems exp-02
            failed on all 4 rejection samples - exp-04 gave these only ~1.2
            solutions each and they were 10% of that corpus
  fresh     up to 2 solutions each for the 11,771 gsm8k-derived problems no run
            has trained on and no run has sampled
  refresh   up to 2 previously-unused solutions for 10,000 already-trained
            problems, so the tail does not pull the model off distribution
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
    ap.add_argument("--hard-per-problem", type=int, default=6)
    ap.add_argument("--fresh-per-problem", type=int, default=2)
    ap.add_argument("--refresh-problems", type=int, default=10000)
    ap.add_argument("--refresh-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-share", type=float, default=0.06)
    ap.add_argument("--max-sol-chars", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # --- which problems fall in which bucket -------------------------------
    sampled = [json.loads(l)["question"] for l in open("data/rft_problems_unseen.jsonl")][:30000]
    solved = set()
    for line in open("data/rft_raw.jsonl"):
        q = question_of(json.loads(line)["prompt"])
        if q:
            solved.add(q)
    hard = {q for q in sampled if q not in solved}

    all_unseen = [json.loads(l)["question"] for l in open("data/rft_problems_unseen.jsonl")]
    fresh = set(all_unseen[30000:])

    trained = [json.loads(l)["question"] for l in open("data/rft_problems_seen.jsonl")]
    rng.shuffle(trained)
    refresh = set(trained[: args.refresh_problems])

    devq = {json.loads(l)["question"] for l in open("data/dev300.jsonl")}
    hard -= devq
    fresh -= devq
    refresh -= devq
    print(f"hard={len(hard)} fresh={len(fresh)} refresh={len(refresh)}")

    # --- solutions already used, so exp-05 rows are all new ----------------
    used: dict[str, set[str]] = defaultdict(set)
    for path in ("data/sft_gsm_clean.jsonl", "data/sft_exp04.jsonl"):
        for line in open(path):
            r = json.loads(line)
            q = question_of(r["prompt"])
            if q:
                used[q].add(r["completion"])

    caps = {}
    for q in hard:
        caps[q] = ("hard", args.hard_per_problem)
    for q in fresh:
        caps[q] = ("fresh", args.fresh_per_problem)
    for q in refresh:
        caps[q] = ("refresh", args.refresh_per_problem)

    kept: dict[str, list[str]] = defaultdict(list)
    counts = defaultdict(int)
    for path in sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
    )):
        df = pq.read_table(
            path, columns=["problem", "generated_solution", "expected_answer", "problem_source"]
        ).to_pandas()
        df = df[df.problem_source.isin(GSM_SOURCES)]
        for problem, sol, ans in zip(df.problem, df.generated_solution, df.expected_answer):
            spec = caps.get(problem)
            if spec is None or len(kept[problem]) >= spec[1]:
                continue
            ans = (ans or "").strip()
            if not INT_ANSWER.match(ans) or len(sol) > args.max_sol_chars:
                continue
            target = clean_solution(sol, ans)
            if target is None or target in used[problem] or target in kept[problem]:
                continue
            kept[problem].append(target)
            counts[spec[0]] += 1
        print(f"{path.split('/')[-1]}: {dict(counts)}", flush=True)

    rows = [(p, t) for p, ts in kept.items() for t in ts]
    rng.shuffle(rows)
    n_few = int(len(rows) * args.fewshot_share)
    few_idx = set(rng.sample(range(len(rows)), n_few))
    with open(args.out, "w") as f:
        for i, (problem, target) in enumerate(rows):
            fs = i in few_idx
            body = MATH_PROMPT_TEMPLATE.format(prompt=problem).strip()
            prompt = f"{FEWSHOT_PREFIX}\n\n{body}" if fs else body
            f.write(json.dumps({"prompt": prompt, "completion": target, "fewshot": fs}) + "\n")
    stats = {"rows": len(rows), "by_bucket": dict(counts), "problems": len(kept),
             "fewshot_rows": n_few, "hard_problems": len(hard), "fresh_problems": len(fresh),
             "refresh_problems": len(refresh)}
    print(json.dumps(stats, indent=2))
    json.dump(stats, open("data/sft_exp05.stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
