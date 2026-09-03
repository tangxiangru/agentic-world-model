#!/usr/bin/env python3
"""Build SFT data for GSM8K from OpenMathInstruct-2 (gsm8k-sourced rows).

Target format is the one the grader reads:
    <reasoning>\n\nANSWER: <number><end_of_turn>
The prompt is the exact user turn inspect_evals/gsm8k renders.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq

STOP = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "

# byte-for-byte from inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^-?\d+(?:\.\d+)?$")
BOXED_RE = re.compile(r"\\boxed\{")


def unbox(text: str) -> str | None:
    """Replace every \\boxed{...} with its contents (brace-matched)."""
    out = text
    for _ in range(10):
        m = BOXED_RE.search(out)
        if m is None:
            return out
        i = m.end()  # just after '{'
        depth = 1
        while i < len(out) and depth:
            if out[i] == "{":
                depth += 1
            elif out[i] == "}":
                depth -= 1
            i += 1
        if depth:
            return None
        out = out[: m.start()] + out[m.end(): i - 1] + out[i:]
    return None


def clean_solution(sol: str, ans: str) -> str | None:
    sol = unbox(sol)
    if sol is None:
        return None
    sol = sol.replace("\\!", "").replace("\\,", " ").strip()
    if not sol:
        return None
    if ANSWER_MARKER in sol or "####" in sol:
        return None
    if "```" in sol or "<end_of_turn>" in sol or "<start_of_turn>" in sol:
        return None
    return sol + "\n\n" + ANSWER_MARKER + ans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-problem", type=int, default=4)
    ap.add_argument("--n", type=int, default=0, help="cap on emitted rows (0 = no cap)")
    ap.add_argument("--sources", default="gsm8k,augmented_gsm8k")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sources = set(args.sources.split(","))
    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
        "469216e3f46f4dacf476b382e192485ea51a143e/data/train_1M-*.parquet"))
    assert files, "no OpenMathInstruct-2 shards found"

    per_problem: dict[str, list[str]] = defaultdict(list)
    answers: dict[str, str] = {}
    seen_sol: set[int] = set()
    stats = defaultdict(int)

    for f in files:
        tbl = pq.read_table(f)
        for rec in tbl.to_pylist():
            stats["seen"] += 1
            if rec["problem_source"] not in sources:
                continue
            stats["src_ok"] += 1
            ans = (rec["expected_answer"] or "").strip()
            if not NUM_RE.match(ans):
                stats["bad_answer"] += 1
                continue
            prob = (rec["problem"] or "").strip()
            if not prob or len(prob) > 1500:
                stats["bad_problem"] += 1
                continue
            tgt = clean_solution(rec["generated_solution"] or "", ans)
            if tgt is None or len(tgt) > 4000:
                stats["bad_solution"] += 1
                continue
            h = hash((prob, tgt))
            if h in seen_sol:
                stats["dup"] += 1
                continue
            seen_sol.add(h)
            if len(per_problem[prob]) >= args.max_per_problem:
                stats["over_cap"] += 1
                continue
            per_problem[prob].append(tgt)
            answers[prob] = ans
            stats["kept"] += 1

    rng = random.Random(args.seed)
    rows = []
    for prob, tgts in per_problem.items():
        for t in tgts:
            rows.append({
                "question": prob,
                "prompt": MATH_PROMPT_TEMPLATE.format(prompt=prob),
                "answer": t + STOP,
                "expected_answer": answers[prob],
            })
    rng.shuffle(rows)
    if args.n:
        rows = rows[: args.n]

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({**stats, "problems": len(per_problem), "emitted": len(rows)}, indent=1))


if __name__ == "__main__":
    main()
