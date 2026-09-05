#!/usr/bin/env python3
"""Build the SFT corpus for GSM8K post-training of gemma-3-4b-pt.

Target format is the one the grader reads (inspect_evals/gsm8k):
  user turn   = MATH_PROMPT_TEMPLATE.format(prompt=question)   (verbatim from the task)
  model turn  = step-by-step reasoning, last line "ANSWER: <number>"
The chat rendering is done at training time with templates/gemma3.jinja, so
every target ends with <end_of_turn>.

Sources (all GSM8K *train*-derived or independent; the test split is never read):
  - openai/gsm8k  train split, original human CoT
  - nvidia/OpenMathInstruct-2, problem_source in {gsm8k, augmented_gsm8k}
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter

from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC = re.compile(r"<<[^>]*>>")
BOXED_SENT = re.compile(r"\s*The final answer is\s*\$?\\boxed\{[^}]*\}\$?\.?\s*$")
BOXED_INLINE = re.compile(r"\\boxed\{([^{}]*)\}")
NUMLIKE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").replace("%", "")
    # >24 digits is not a grade-school answer and overflows float()
    if not NUMLIKE.match(a) or len(a) > 24:
        return None
    if a.endswith(".0"):
        a = a[:-2]
    try:
        f = float(a)
        if not math.isfinite(f):
            return None
        if f == int(f):
            return str(int(f))
    except (ValueError, OverflowError):
        return None
    return a


def clean_solution(sol: str) -> str:
    sol = CALC.sub("", sol)
    sol = BOXED_SENT.sub("", sol)
    sol = BOXED_INLINE.sub(r"\1", sol)
    sol = re.sub(r"\n{3,}", "\n\n", sol).strip()
    return sol


def make_row(question: str, solution: str, answer: str) -> dict | None:
    q = question.strip()
    sol = clean_solution(solution)
    if not q or not sol:
        return None
    # the grader reads the LAST number: nothing may follow the ANSWER line
    # the stop token the grading template terminates model turns with
    target = f"{sol}\n\nANSWER: {answer}<end_of_turn>"
    return {
        "prompt": MATH_PROMPT_TEMPLATE.format(prompt=q),
        "completion": target,
        "question": q,
        "answer": answer,
    }


def sample_to_fewshot(question: str, reasoning: str, answer: str) -> str:
    """Exactly the exemplar shape the grader's system message uses."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--n-omi", type=int, default=80000)
    ap.add_argument("--omi-split", default="train")
    ap.add_argument("--max-per-problem", type=int, default=1)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []
    stats: Counter = Counter()

    # ---- 1. gsm8k train, original human chain of thought -------------------
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for rec in gsm:
        sol, _, ans = rec["answer"].rpartition("####")
        a = norm_answer(ans)
        if a is None:
            stats["gsm8k_bad_answer"] += 1
            continue
        r = make_row(rec["question"], sol, a)
        if r is None:
            continue
        r["source"] = "gsm8k_train"
        for _ in range(args.gsm8k_repeat):
            rows.append(dict(r))
        stats["gsm8k_train"] += args.gsm8k_repeat

    # ---- 2. OpenMathInstruct-2, gsm8k-derived problems ---------------------
    omi = load_dataset("nvidia/OpenMathInstruct-2", split=args.omi_split)
    omi = omi.filter(
        lambda x: x["problem_source"] in ("gsm8k", "augmented_gsm8k"),
        num_proc=16,
    )
    print("omi gsm8k-family rows:", len(omi), flush=True)
    idx = list(range(len(omi)))
    rng.shuffle(idx)
    per_problem: Counter = Counter()
    kept = 0
    for i in idx:
        if kept >= args.n_omi:
            break
        rec = omi[i]
        q = rec["problem"].strip()
        if per_problem[q] >= args.max_per_problem:
            continue
        a = norm_answer(rec["expected_answer"])
        if a is None:
            stats["omi_non_numeric"] += 1
            continue
        sol = rec["generated_solution"]
        if "\\boxed" in clean_solution(sol):
            stats["omi_leftover_boxed"] += 1
            continue
        r = make_row(q, sol, a)
        if r is None:
            continue
        r["source"] = "openmathinstruct2"
        rows.append(r)
        per_problem[q] += 1
        kept += 1
        stats["omi"] += 1

    # ---- 3. few-shot conditioning on a slice of rows -----------------------
    # The grader always prepends a 10-shot system message (gsm8k TRAIN
    # exemplars, seed 42).  Training a slice with variable-length exemplar
    # prefixes -- drawn from the train split, never the test split -- keeps the
    # model robust to that prompt shape instead of only the zero-shot one.
    pool = [
        (rec["question"].strip(),
         CALC.sub("", rec["answer"].rpartition("####")[0]).strip(),
         norm_answer(rec["answer"].rpartition("####")[2]))
        for rec in gsm
    ]
    pool = [p for p in pool if p[2] is not None]
    for r in rows:
        if rng.random() >= args.fewshot_frac:
            r["system"] = None
            continue
        k = rng.randint(2, 5)
        shots = rng.sample(pool, k)
        r["system"] = "\n\n".join(sample_to_fewshot(*s) for s in shots)
        stats["fewshot_rows"] += 1

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(rows), "rows to", args.out)
    print(stats)


if __name__ == "__main__":
    main()
