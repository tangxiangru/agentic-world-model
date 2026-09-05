#!/usr/bin/env python3
"""Build SFT data for GSM8K from OpenMathInstruct-2 (gsm8k-derived subsets only).

Output JSONL rows: {question, answer, prompt, completion}
  prompt     - rendered exactly as the grader renders it (templates/gemma3.jinja)
  completion - solution body + "ANSWER: N" + <end_of_turn>
`question`/`answer` are kept so ../contamination_check.py can read the row.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"
SHARDS = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
NUMLIKE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def unboxed(sol: str) -> str | None:
    """Replace the single \\boxed{X} with X; reject rows with 0 or >1 boxes."""
    hits = BOXED.findall(sol)
    if len(hits) != 1:
        return None
    return BOXED.sub(lambda m: m.group(1), sol)


def norm_num(s: str) -> str:
    return s.replace(",", "").replace("$", "").strip()


def build_prompt(tok, question: str) -> str:
    msg = [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=question)}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_omi2.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=1)
    ap.add_argument("--max-rows", type=int, default=250000)
    ap.add_argument("--max-completion-tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    tok.chat_template = open(TEMPLATE).read()

    files = sorted(glob.glob(SHARDS))
    assert files, "no OpenMathInstruct-2 shards found"
    rng = random.Random(args.seed)

    rows, per_problem = [], {}
    kept_src = {"gsm8k": 0, "augmented_gsm8k": 0}
    for f in files:
        t = pq.read_table(
            f, columns=["problem", "generated_solution", "expected_answer", "problem_source"]
        ).to_pylist()
        for r in t:
            src = r["problem_source"]
            if src not in ("gsm8k", "augmented_gsm8k"):
                continue
            ans = norm_num(r["expected_answer"] or "")
            if not NUMLIKE.match(ans):
                continue
            body = unboxed(r["generated_solution"] or "")
            if body is None:
                continue
            # the boxed value must be the answer we assert
            if norm_num(BOXED.findall(r["generated_solution"])[0]) != ans:
                continue
            q = r["problem"].strip()
            if per_problem.get(q, 0) >= args.max_per_problem:
                continue
            body = body.strip()
            if "ANSWER:" in body or "####" in body:
                continue
            per_problem[q] = per_problem.get(q, 0) + 1
            kept_src[src] += 1
            rows.append((q, body, ans))
    print(f"candidate rows: {len(rows)}  by source: {kept_src}")

    rng.shuffle(rows)
    rows = rows[: args.max_rows * 2]

    out = []
    lens = []
    dropped_long = 0
    for q, body, ans in rows:
        completion = f"{body}\n\nANSWER: {ans}<end_of_turn>"
        n_c = len(tok(completion, add_special_tokens=False)["input_ids"])
        if n_c > args.max_completion_tokens:
            dropped_long += 1
            continue
        prompt = build_prompt(tok, q)
        n_p = len(tok(prompt, add_special_tokens=False)["input_ids"])
        lens.append(n_p + n_c)
        out.append({"question": q, "answer": f"{body}\n#### {ans}", "prompt": prompt,
                    "completion": completion})
        if len(out) >= args.max_rows:
            break

    lens.sort()
    print(f"kept {len(out)} rows (dropped {dropped_long} over-long completions)")
    print(f"total token length: p50={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} max={lens[-1]}")
    with open(args.out, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
