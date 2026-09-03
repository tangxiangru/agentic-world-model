#!/usr/bin/env python3
"""Build SFT data in the exact format the GSM8K grader scores.

The grader (inspect_evals/gsm8k) wraps every question in MATH_PROMPT_TEMPLATE and
scores with match(numeric=True, location="end"), which reads the LAST number in the
completion. So every training target ends with a single "ANSWER: <number>" line and
nothing after it.

Sources (both GSM8K *train*-derived, never test):
  * nvidia/OpenMathInstruct-2, problem_source in {gsm8k, augmented_gsm8k}
  * openai/gsm8k main/train, with the <<...>> calculator annotations stripped

Output columns: prompt, completion, answer, source
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from typing import Iterator

import pyarrow.parquet as pq

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

OMI2_GLOBS = [
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet",
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_2M-*.parquet",
]
GSM8K_TRAIN = "/home/ben/task/data/gsm8k_train_raw.jsonl"

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
CALC = re.compile(r"<<[^>]*>>")
NUMLIKE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def normalize_answer(a: str) -> str | None:
    """Return the answer as the grader would compare it, or None if not numeric."""
    a = a.strip().replace("$", "").replace(",", "").rstrip(".")
    a = a.replace("\\%", "").replace("%", "").strip()
    if not NUMLIKE.match(a):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    return a


def last_number(text: str) -> str | None:
    """Mirror inspect's match(numeric=True, location='end'): the last whitespace token
    that is numeric after punctuation stripping."""
    words = re.split(r"\s+", text.strip())
    for w in reversed(words):
        c = w.strip().replace("$", "").replace(",", "").rstrip(".").rstrip("%")
        if c and NUMLIKE.match(c):
            return c.rstrip(".")
    return None


def make_row(problem: str, solution: str, answer: str, source: str) -> dict | None:
    ans = normalize_answer(answer)
    if ans is None:
        return None
    body = BOXED.sub(r"\1", solution).strip()
    body = CALC.sub("", body)
    # kill any trailing "#### N" marker so there is exactly one answer marker
    body = re.sub(r"####\s*[-\d,\.]+\s*$", "", body).strip()
    if not body:
        return None
    completion = f"{body}\nANSWER: {ans}"
    if last_number(completion) != ans:
        return None
    if completion.count("ANSWER:") != 1:
        return None
    return {
        "problem": problem.strip(),
        "prompt": MATH_PROMPT_TEMPLATE.format(prompt=problem.strip()),
        "completion": completion,
        "answer": ans,
        "source": source,
    }


def iter_omi2() -> Iterator[dict]:
    files = sorted(f for g in OMI2_GLOBS for f in glob.glob(g))
    assert files, "OpenMathInstruct-2 parquet shards not found"
    keep = {"gsm8k", "augmented_gsm8k"}
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=20000):
            for r in batch.to_pylist():
                if r["problem_source"] not in keep:
                    continue
                if r["generated_solution"].count("\\boxed{") != 1:
                    continue
                row = make_row(
                    r["problem"], r["generated_solution"], r["expected_answer"],
                    f"omi2:{r['problem_source']}",
                )
                if row is not None:
                    yield row


def iter_gsm8k_train() -> Iterator[dict]:
    with open(GSM8K_TRAIN) as fh:
        for line in fh:
            r = json.loads(line)
            body, _, ans = r["answer"].rpartition("####")
            row = make_row(r["question"], body, ans, "gsm8k_train")
            if row is not None:
                yield row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/pool.jsonl")
    ap.add_argument("--max-chars", type=int, default=3000,
                    help="drop solutions longer than this (rough token guard)")
    args = ap.parse_args()

    seen_problems: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}
    dropped_long = 0
    n = 0
    with open(args.out, "w") as out:
        for row in list(iter_gsm8k_train()) + list(iter_omi2()):
            if len(row["completion"]) > args.max_chars or len(row["prompt"]) > args.max_chars:
                dropped_long += 1
                continue
            key = row["prompt"]
            pair = (key, row["completion"])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            # at most 2 solutions per distinct problem, to keep diversity of problems high
            c = counts.get(key, 0)
            if c >= 2:
                continue
            counts[key] = c + 1
            seen_problems.add(key)
            out.write(json.dumps(row) + "\n")
            n += 1
    print(f"wrote {n} rows to {args.out}; distinct problems {len(seen_problems)}; dropped_long {dropped_long}")


if __name__ == "__main__":
    main()
