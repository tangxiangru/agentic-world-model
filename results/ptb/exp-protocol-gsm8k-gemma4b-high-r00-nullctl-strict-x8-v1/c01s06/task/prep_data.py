#!/usr/bin/env python3
"""Build the SFT dataset for GSM8K in the exact format the evaluation harness uses."""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq
from datasets import load_dataset

# Exactly the template used by inspect_evals/gsm8k
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

OMI_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"

NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def clean_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").replace("\\%", "").replace("%", "")
    a = a.strip()
    if not NUM_RE.match(a):
        return None
    # normalise "12.0" -> "12"
    if "." in a:
        f = float(a)
        if f == int(f):
            a = str(int(f))
    return a


def strip_boxed(sol: str, ans: str) -> str | None:
    """Replace the final \\boxed{...} with plain text."""
    idx = sol.rfind("\\boxed{")
    if idx == -1:
        return None
    # find matching brace
    i = idx + len("\\boxed{")
    depth = 1
    while i < len(sol) and depth:
        if sol[i] == "{":
            depth += 1
        elif sol[i] == "}":
            depth -= 1
        i += 1
    if depth:
        return None
    inner = sol[idx + len("\\boxed{"): i - 1]
    return sol[:idx] + inner + sol[i:]


def build_target(body: str, ans: str) -> str:
    body = body.strip()
    # drop a trailing sentence fragment that is now dangling punctuation only
    return f"{body}\n\nANSWER: {ans}"


def gsm8k_train_examples():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for r in ds:
        q = r["question"].strip()
        sol, ans = r["answer"].split("####")
        ans = clean_answer(ans)
        if ans is None:
            continue
        sol = re.sub(r"<<[^>]*>>", "", sol).strip()
        out.append({"question": q, "solution": build_target(sol, ans), "answer": ans,
                    "src": "gsm8k_orig"})
    return out


def gsm8k_fewshot_raw():
    """The reasoning strings used by the harness' few-shot block (with <<>> kept)."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for r in ds:
        sol, ans = r["answer"].split("####")
        ans = ans.strip()
        out.append((r["question"].strip(), sol.strip(), ans))
    return out


def omi_examples(max_per_problem: int, limit: int | None):
    files = sorted(glob.glob(OMI_GLOB))
    by_problem: dict[str, list] = defaultdict(list)
    for f in files:
        tbl = pq.read_table(f)
        srcs = tbl.column("problem_source").to_pylist()
        probs = tbl.column("problem").to_pylist()
        sols = tbl.column("generated_solution").to_pylist()
        answers = tbl.column("expected_answer").to_pylist()
        for s, p, sol, a in zip(srcs, probs, sols, answers):
            if s not in ("gsm8k", "augmented_gsm8k"):
                continue
            ans = clean_answer(a)
            if ans is None:
                continue
            body = strip_boxed(sol, ans)
            if body is None:
                continue
            if len(body) > 4000 or len(p) > 2000:
                continue
            by_problem[p.strip()].append((body, ans, s))
    rng = random.Random(0)
    out = []
    for p, cands in by_problem.items():
        rng.shuffle(cands)
        seen = set()
        kept = 0
        for body, ans, s in cands:
            key = body[:200]
            if key in seen:
                continue
            seen.add(key)
            out.append({"question": p, "solution": build_target(body, ans),
                        "answer": ans, "src": "omi_" + s})
            kept += 1
            if kept >= max_per_problem:
                break
    rng.shuffle(out)
    if limit:
        out = out[:limit]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--limit-omi", type=int, default=90000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.18)
    ap.add_argument("--gsm-orig-repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    pool = omi_examples(args.max_per_problem, args.limit_omi)
    orig = gsm8k_train_examples()
    for _ in range(args.gsm_orig_repeat):
        pool.extend(orig)
    rng.shuffle(pool)

    fewshot_pool = gsm8k_fewshot_raw()

    with open(args.out, "w") as f:
        for ex in pool:
            prompt_body = MATH_PROMPT_TEMPLATE.format(prompt=ex["question"])
            prefix = ""
            if rng.random() < args.fewshot_frac:
                k = rng.randint(1, 5)
                shots = rng.sample(fewshot_pool, k)
                blocks = [f"{q}\n\nReasoning:\n{s}\n\nANSWER: {a}" for q, s, a in shots]
                prefix = "\n\n".join(blocks) + "\n\n"
            f.write(json.dumps({
                "prompt": prefix + prompt_body,
                "completion": ex["solution"],
                "src": ex["src"],
            }) + "\n")
    print(f"wrote {len(pool)} examples to {args.out}")


if __name__ == "__main__":
    main()
