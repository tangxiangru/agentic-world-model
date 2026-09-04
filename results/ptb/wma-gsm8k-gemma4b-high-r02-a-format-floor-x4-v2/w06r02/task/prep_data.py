#!/usr/bin/env python3
"""Build SFT data for GSM8K in the exact format the inspect_evals/gsm8k grader expects.

Sources: nvidia/OpenMathInstruct-2 (train_1M), gsm8k-derived rows only.
Those rows are augmentations of the GSM8K *train* split; the test split is never touched.

Target format (one answer marker, ends with the grader's stop token when rendered):
    <reasoning ...>

    ANSWER: <number>
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

from datasets import load_dataset

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"^-?\d{1,12}(\.\d{1,6})?$")


def is_plain_number(s: str) -> bool:
    return bool(NUM_RE.match(s.strip().replace(",", "").replace("$", "")))


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{X} with X (balanced-brace aware)."""
    out = []
    i = 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k = j + len("\\boxed{")
        depth = 1
        while k < len(text) and depth:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        out.append(text[j + len("\\boxed{"): k - 1])
        i = k
    return "".join(out)


def clean_solution(sol: str) -> str:
    sol = strip_boxed(sol)
    # drop trailing latex leftovers of the form "\[ 12 \]" that end the solution
    sol = sol.strip()
    return sol


def build(args):
    rng = random.Random(0)
    ds = load_dataset("nvidia/OpenMathInstruct-2", split=args.split)
    ds = ds.filter(
        lambda r: r["problem_source"] in ("gsm8k", "augmented_gsm8k"),
        num_proc=8,
    )
    print("gsm8k-derived rows:", len(ds))

    by_problem = defaultdict(list)
    for rec in ds:
        ans = rec["expected_answer"].strip()
        if not is_plain_number(ans):
            continue
        by_problem[rec["problem"].strip()].append(rec)
    print("unique problems:", len(by_problem))

    rows = []
    for prob, recs in by_problem.items():
        rng.shuffle(recs)
        for rec in recs[: args.per_problem]:
            ans = rec["expected_answer"].strip().replace(",", "")
            sol = clean_solution(rec["generated_solution"])
            if not sol or len(sol) > args.max_chars:
                continue
            # the last line must be the single answer marker
            target = f"{sol}\n\nANSWER: {ans}"
            if target.count("ANSWER:") != 1:
                continue
            rows.append(
                {
                    "prompt": PROMPT_TEMPLATE.format(prompt=prob),
                    "completion": target,
                    "question": prob,
                    "answer": ans,
                    "source": rec["problem_source"],
                }
            )
    rng.shuffle(rows)
    if args.n and args.n < len(rows):
        rows = rows[: args.n]
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(rows), "rows to", args.out)
    lens = sorted(len(r["prompt"]) + len(r["completion"]) for r in rows)
    print("chars p50/p95/p99/max:", lens[len(lens) // 2], lens[int(0.95 * len(lens))],
          lens[int(0.99 * len(lens))], lens[-1])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train_1M")
    p.add_argument("--per-problem", type=int, default=2)
    p.add_argument("--max-chars", type=int, default=3000)
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--out", default="data/sft_gsm8k.jsonl")
    build(p.parse_args())
