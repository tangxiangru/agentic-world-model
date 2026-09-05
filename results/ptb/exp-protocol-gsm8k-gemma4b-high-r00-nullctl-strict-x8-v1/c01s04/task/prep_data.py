#!/usr/bin/env python3
"""Build SFT dataset for GSM8K in the exact format used by evaluate.py."""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

OMI_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{...} with its content (handles nested braces)."""
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
                if depth == 0:
                    break
            k += 1
        out.append(text[j + len("\\boxed{"): k])
        i = k + 1
    return "".join(out)


NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def is_plain_number(s: str) -> bool:
    s = s.strip().replace(",", "")
    if s.endswith(".0"):
        s = s[:-2]
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", s))


def norm_number(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return ("%f" % f).rstrip("0").rstrip(".")
    except Exception:
        return s


def clean_gsm8k_reasoning(ans: str) -> tuple[str, str]:
    body, final = ans.split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return body, final.strip().replace(",", "")


def build_example(question: str, reasoning: str, answer: str) -> dict:
    reasoning = reasoning.strip()
    completion = f"{reasoning}\n\nANSWER: {answer}"
    return {
        "prompt": MATH_PROMPT_TEMPLATE.format(prompt=question.strip()),
        "completion": completion,
        "question": question.strip(),
        "answer": answer,
    }


def load_openmath(max_sol_per_problem: int, sources: set[str], rng: random.Random):
    files = sorted(glob.glob(OMI_GLOB))
    by_problem = defaultdict(list)
    for f in files:
        df = pq.read_table(f, columns=["problem", "generated_solution", "expected_answer", "problem_source"]).to_pandas()
        df = df[df.problem_source.isin(sources)]
        for prob, sol, ans in zip(df.problem, df.generated_solution, df.expected_answer):
            if not is_plain_number(ans):
                continue
            by_problem[prob].append((sol, ans))
        del df
    out = []
    for prob, sols in by_problem.items():
        rng.shuffle(sols)
        for sol, ans in sols[:max_sol_per_problem]:
            sol = strip_boxed(sol).strip()
            # solution must be self-consistent: last number in text == answer
            nums = NUM_RE.findall(sol)
            if not nums or norm_number(nums[-1]) != norm_number(ans):
                continue
            if "\\[" in sol or "\\(" in sol or "```" in sol:
                continue
            if len(sol) > 3000 or len(sol) < 30:
                continue
            out.append(build_example(prob, sol, norm_number(ans)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--max-sol", type=int, default=2)
    ap.add_argument("--gsm-repeat", type=int, default=2, help="times to repeat original gsm8k train gold CoT")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main")["train"]
    gold = []
    for rec in gsm:
        body, final = clean_gsm8k_reasoning(rec["answer"])
        gold.append(build_example(rec["question"], body, final))
    print(f"gsm8k gold: {len(gold)}")

    omi = load_openmath(args.max_sol, {"gsm8k", "augmented_gsm8k"}, rng)
    print(f"openmathinstruct gsm-family: {len(omi)}")

    data = gold * args.gsm_repeat + omi
    rng.shuffle(data)
    if args.limit:
        data = data[: args.limit]

    with open(args.out, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    print(f"wrote {len(data)} -> {args.out}")


if __name__ == "__main__":
    main()
