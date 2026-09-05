#!/usr/bin/env python3
"""Build the SFT corpus from OpenMathInstruct-2 (gsm8k-derived rows only).

Output: jsonl with {prompt, completion, answer, source} where
  prompt     = the exact user-turn content the grader sends (MATH_PROMPT_TEMPLATE)
  completion = reasoning + "\n\nANSWER: <n>"   (the grader reads the last number)

Nothing here touches the GSM8K test split: OpenMathInstruct-2 is built from the
GSM8K/MATH *train* splits. The contamination checker is run separately on the
output.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import re

import pyarrow.parquet as pq

STOP_TOKEN = "<end_of_turn>"

# byte-for-byte the template inspect_evals/gsm8k applies to every sample
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
NUMLIKE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def unbox(text: str) -> str:
    """Replace \\boxed{x} with x, and drop leftover latex math delimiters."""
    prev = None
    while prev != text:
        prev = text
        text = BOXED.sub(r"\1", text)
    text = text.replace("\\[", "").replace("\\]", "")
    return text


def norm_num(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip("%")
    if not NUMLIKE.match(s.replace(",", "")):
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", default="gsm8k,augmented_gsm8k")
    ap.add_argument("--split-glob", default="train_1M-*")
    ap.add_argument("--exclude-problems-from", default=None,
                    help="jsonl whose 'problem' values are excluded (keeps rounds disjoint)")
    ap.add_argument("--exclude-pairs-from", default=None,
                    help="jsonl whose (problem, completion) pairs are excluded")
    args = ap.parse_args()

    keep_sources = set(args.sources.split(","))
    files = sorted(
        glob.glob(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
            f"snapshots/*/data/{args.split_glob}.parquet"
        )
    )
    assert files, "OpenMathInstruct-2 train_1M not downloaded"

    excluded = set()
    seen_pairs_prev: set[tuple[str, str]] = set()
    if args.exclude_problems_from:
        with open(args.exclude_problems_from) as f:
            for line in f:
                excluded.add(json.loads(line)["problem"])
        print(f"excluding {len(excluded)} problems seen in a previous round")

    if args.exclude_pairs_from:
        with open(args.exclude_pairs_from) as f:
            for line in f:
                r = json.loads(line)
                seen_pairs_prev.add((r["problem"], r["completion"]))
        print(f"excluding {len(seen_pairs_prev)} (problem, completion) pairs from a previous round")

    rows = []
    stats = collections.Counter()
    for f in files:
        tbl = pq.read_table(
            f, columns=["problem", "generated_solution", "expected_answer", "problem_source"]
        )
        for r in tbl.to_pylist():
            stats["seen"] += 1
            if r["problem_source"] not in keep_sources:
                continue
            stats["source_ok"] += 1
            if r["problem"].strip() in excluded:
                stats["drop_excluded_problem"] += 1
                continue
            ans = norm_num(r["expected_answer"])
            if ans is None:
                stats["drop_nonnumeric_answer"] += 1
                continue
            sol = unbox(r["generated_solution"]).strip()
            if "####" in sol:
                stats["drop_hash_marker"] += 1
                continue
            if "ANSWER:" in sol.upper():
                stats["drop_answer_marker_in_body"] += 1
                continue
            # the boxed value must agree with the expected answer
            boxed = BOXED.findall(r["generated_solution"])
            if boxed:
                b = norm_num(boxed[-1])
                if b is None or b != ans:
                    stats["drop_boxed_mismatch"] += 1
                    continue
            else:
                stats["drop_no_boxed"] += 1
                continue
            if len(sol) > args.max_chars or len(sol) < 40:
                stats["drop_length"] += 1
                continue
            completion = f"{sol}\n\nANSWER: {ans}{STOP_TOKEN}"
            rows.append(
                {
                    "problem": r["problem"].strip(),
                    "completion": completion,
                    "answer": ans,
                    "source": r["problem_source"],
                }
            )
            stats["kept_pre_dedup"] += 1

    # cap solutions per unique problem, drop exact duplicate completions
    random.Random(args.seed).shuffle(rows)
    per_problem: dict[str, int] = collections.defaultdict(int)
    seen_pairs: set[tuple[str, str]] = set(seen_pairs_prev)
    out = []
    for r in rows:
        key = r["problem"]
        if per_problem[key] >= args.max_per_problem:
            stats["drop_per_problem_cap"] += 1
            continue
        pair = (key, r["completion"])
        if pair in seen_pairs:
            stats["drop_exact_dup"] += 1
            continue
        seen_pairs.add(pair)
        per_problem[key] += 1
        r["prompt"] = MATH_PROMPT_TEMPLATE.format(prompt=key)
        out.append(r)

    random.Random(args.seed + 1).shuffle(out)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats["final"] = len(out)
    stats["unique_problems"] = len(per_problem)
    print(json.dumps(dict(sorted(stats.items())), indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
