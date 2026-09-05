#!/usr/bin/env python3
"""Build SFT data for GSM8K from OpenMathInstruct-2 (gsm8k / augmented_gsm8k subsets).

Target format is the one the grader reads:
  - prompt: the exact inspect_evals gsm8k user turn (MATH_PROMPT_TEMPLATE)
  - completion: the chain of thought, then a final line "ANSWER: <number>"
No other answer marker survives (\\boxed{} is unwrapped), so the grader's
"last number in the completion" rule and the "ANSWER:" line agree.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

from datasets import load_dataset

# copied verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def unwrap_boxed(text: str) -> str:
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
    out = []
    i = 0
    while True:
        j = text.find("\\boxed{", i)
        if j == -1:
            out.append(text[i:])
            return "".join(out)
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


def clean_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").replace("%", "")
    if not re.fullmatch(r"-?\d+(\.\d+)?", a):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    if "." in a:
        a = a.rstrip("0").rstrip(".")
    return a


def last_number(text: str) -> str | None:
    """Reproduce the grader's extraction: last whitespace-token that is numeric."""
    words = re.split(r"\s+", text.strip())
    for w in reversed(words):
        w2 = re.sub(r"[$,£€*_]", "", w)
        w2 = re.sub(r"\.(?=\s|$|\D)", "", w2)
        if w2.replace(".", "").isnumeric():
            return w2
    return None


def build_completion(solution: str, answer: str) -> str | None:
    body = unwrap_boxed(solution).strip()
    # kill leftover markers that would confuse a reader / a second answer format
    if "####" in body:
        body = body.split("####")[0].strip()
    if not body:
        return None
    comp = f"{body}\n\nANSWER: {answer}"
    # single answer marker, and the grader's last-number rule must land on `answer`
    if comp.count("ANSWER:") != 1:
        return None
    if last_number(comp) != answer:
        return None
    return comp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_omi2_gsm8k.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--n", type=int, default=64000)
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    ds = ds.filter(
        lambda x: x["problem_source"] in ("gsm8k", "augmented_gsm8k"), num_proc=8
    )
    print("gsm8k-sourced rows:", len(ds))

    by_problem: dict[str, list[dict]] = defaultdict(list)
    dropped = defaultdict(int)
    for row in ds:
        ans = clean_answer(row["expected_answer"])
        if ans is None:
            dropped["non_numeric_answer"] += 1
            continue
        sol = row["generated_solution"]
        if not (args.min_chars <= len(sol) <= args.max_chars):
            dropped["length"] += 1
            continue
        comp = build_completion(sol, ans)
        if comp is None:
            dropped["format"] += 1
            continue
        by_problem[row["problem"]].append(
            {
                "prompt": MATH_PROMPT_TEMPLATE.format(prompt=row["problem"].strip()),
                "completion": comp,
                "answer": ans,
                "source": row["problem_source"],
                "problem": row["problem"],
            }
        )

    print("distinct problems kept:", len(by_problem), "dropped:", dict(dropped))

    # prefer real gsm8k-train problems, then augmented; up to k solutions each,
    # preferring shorter-but-not-trivial solutions for a tighter style
    real, aug = [], []
    for prob, rows in by_problem.items():
        rng.shuffle(rows)
        keep = rows[: args.max_per_problem]
        (real if keep[0]["source"] == "gsm8k" else aug).append(keep)

    rng.shuffle(real)
    rng.shuffle(aug)
    out_rows: list[dict] = []
    for group in real:
        out_rows.extend(group)
    for group in aug:
        if len(out_rows) >= args.n:
            break
        out_rows.extend(group)
    out_rows = out_rows[: args.n]
    rng.shuffle(out_rows)

    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    n_real = sum(1 for r in out_rows if r["source"] == "gsm8k")
    print(f"wrote {len(out_rows)} rows to {args.out} ({n_real} from gsm8k-train problems)")
    print("--- example ---")
    print(out_rows[0]["prompt"])
    print("--- completion ---")
    print(out_rows[0]["completion"])


if __name__ == "__main__":
    main()
