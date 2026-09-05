#!/usr/bin/env python3
"""Build SFT data for GSM8K from OpenMathInstruct-2 (GSM8K-train-derived) + GSM8K train."""
from __future__ import annotations

import argparse
import glob
import json
import random
import re

import pyarrow.parquet as pq

PROMPT = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOXED_RE = re.compile(r"\\boxed\s*\{")


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{...} with its contents."""
    out = []
    i = 0
    while True:
        m = BOXED_RE.search(text, i)
        if m is None:
            out.append(text[i:])
            break
        out.append(text[i : m.start()])
        j = m.end()
        depth = 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        out.append(text[m.end() : j - 1])
        i = j
    return "".join(out)


def clean_solution(sol: str) -> str:
    sol = strip_boxed(sol)
    # drop stray latex display wrappers that add no value
    sol = sol.replace("\\[", "").replace("\\]", "")
    sol = re.sub(r"[ \t]+\n", "\n", sol)
    sol = re.sub(r"\n{3,}", "\n\n", sol)
    return sol.strip()


NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def is_plain_number(s: str) -> bool:
    s = s.strip().replace(",", "").replace("$", "")
    if s.startswith("-"):
        s = s[1:]
    return bool(re.fullmatch(r"\d+(\.\d+)?", s))


def norm_answer(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def load_omi2(sources, max_per_source):
    files = sorted(
        glob.glob(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
            "snapshots/*/data/train_1M-*.parquet"
        )
    )
    assert files, "OpenMathInstruct-2 parquet files not found"
    rows = []
    counts = {s: 0 for s in sources}
    for f in files:
        t = pq.read_table(f)
        d = t.to_pydict()
        for prob, sol, ans, src in zip(
            d["problem"], d["generated_solution"], d["expected_answer"], d["problem_source"]
        ):
            if src not in counts:
                continue
            if counts[src] >= max_per_source.get(src, 0):
                continue
            counts[src] += 1
            rows.append((prob, sol, ans, src))
    print("loaded from OMI2:", counts)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--n-aug-gsm8k", type=int, default=200000)
    ap.add_argument("--n-gsm8k", type=int, default=200000)
    ap.add_argument("--n-aug-math", type=int, default=0)
    ap.add_argument("--n-math", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    max_per_source = {
        "augmented_gsm8k": args.n_aug_gsm8k,
        "gsm8k": args.n_gsm8k,
        "augmented_math": args.n_aug_math,
        "math": args.n_math,
    }
    sources = [k for k, v in max_per_source.items() if v > 0]
    rows = load_omi2(sources, max_per_source)

    kept, dropped = [], 0
    seen = set()
    for prob, sol, ans, src in rows:
        ans = norm_answer(ans)
        if not is_plain_number(ans) and src in ("gsm8k", "augmented_gsm8k"):
            dropped += 1
            continue
        sol_c = clean_solution(sol)
        if not sol_c or len(sol_c) < 20:
            dropped += 1
            continue
        # answer must be the final number of the completion -> we append it explicitly
        completion = f"{sol_c}\n\nANSWER: {ans}"
        key = (prob.strip(), ans)
        if key in seen and src != "gsm8k":
            # allow multiple solutions per problem, but cap duplicates of identical text
            pass
        seen.add(key)
        kept.append(
            {
                "prompt": PROMPT.format(prompt=prob.strip()),
                "completion": completion,
                "source": src,
                "answer": ans,
                "question": prob.strip(),
            }
        )
    print(f"kept={len(kept)} dropped={dropped}")

    rng.shuffle(kept)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
