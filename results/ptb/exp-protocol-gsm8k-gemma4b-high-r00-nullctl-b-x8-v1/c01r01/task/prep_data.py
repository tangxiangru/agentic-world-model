#!/usr/bin/env python3
"""Build the SFT dataset in exactly the format the inspect_evals GSM8K task uses."""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

OMI_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"

BOXED_RE = re.compile(r"\\boxed\{")


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{...} with its inner content (brace-balanced)."""
    while True:
        m = BOXED_RE.search(text)
        if m is None:
            return text
        start = m.end()  # first char inside the brace
        depth = 1
        i = start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth:  # unbalanced -> give up
            return text
        text = text[: m.start()] + text[start : i - 1] + text[i:]


def norm_num(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    return s


def clean_gsm_answer(ans: str) -> tuple[str, str]:
    """Original GSM8K answer -> (reasoning without calculator annotations, final)."""
    reasoning, _, final = ans.partition("####")
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning).strip()
    return reasoning, norm_num(final)


def make_record(question: str, solution: str, answer: str) -> dict | None:
    solution = solution.strip()
    if not solution:
        return None
    answer = answer.strip()
    # Drop a trailing "The answer is ..." style sentence, we append our own line.
    solution = re.sub(
        r"\n?(?:The (?:final )?answer is[: ].*|#### .*)\s*$", "", solution
    ).strip()
    if not solution:
        return None
    body = f"{solution}\n\nANSWER: {answer}"
    return {
        "messages": [
            {"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=question.strip())},
            {"role": "assistant", "content": body},
        ],
        "question": question.strip(),
        "answer": answer,
    }


def load_omi(max_per_problem_gsm: int, max_per_problem_math: int, math_frac_cap: int):
    files = sorted(glob.glob(OMI_GLOB))
    assert files, "OpenMathInstruct-2 shards not found"
    by_problem_gsm: dict[str, list] = defaultdict(list)
    by_problem_math: dict[str, list] = defaultdict(list)
    for f in files:
        t = pq.read_table(f, columns=["problem", "generated_solution", "expected_answer", "problem_source"])
        cols = t.to_pydict()
        for p, s, a, src in zip(
            cols["problem"], cols["generated_solution"], cols["expected_answer"], cols["problem_source"]
        ):
            if src in ("gsm8k", "augmented_gsm8k"):
                bucket = by_problem_gsm
                cap = max_per_problem_gsm
            else:
                bucket = by_problem_math
                cap = max_per_problem_math
            if len(bucket[p]) < cap:
                bucket[p].append((s, a))
        del t, cols
    out_gsm, out_math = [], []
    for bucket, out in ((by_problem_gsm, out_gsm), (by_problem_math, out_math)):
        for p, sols in bucket.items():
            for s, a in sols:
                r = make_record(p, strip_boxed(s), a)
                if r is not None:
                    out.append(r)
    rng = random.Random(1234)
    rng.shuffle(out_math)
    return out_gsm, out_math[:math_frac_cap]


def load_gsm_train():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for row in ds:
        reasoning, final = clean_gsm_answer(row["answer"])
        r = make_record(row["question"], reasoning, final)
        if r is not None:
            out.append(r)
    return out


def token_len_ok(recs, tokenizer, max_len):
    keep = []
    texts = [r["messages"][0]["content"] + r["messages"][1]["content"] for r in recs]
    enc = tokenizer(texts, add_special_tokens=False)["input_ids"]
    for r, ids in zip(recs, enc):
        if len(ids) + 8 <= max_len:
            keep.append(r)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/sft_v1.jsonl")
    ap.add_argument("--max-per-problem-gsm", type=int, default=4)
    ap.add_argument("--max-per-problem-math", type=int, default=1)
    ap.add_argument("--math-cap", type=int, default=20000)
    ap.add_argument("--gsm-repeat", type=int, default=2, help="repeats of the original GSM8K train CoT")
    ap.add_argument("--total-cap", type=int, default=200000)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    print("loading OpenMathInstruct-2 ...", flush=True)
    omi_gsm, omi_math = load_omi(args.max_per_problem_gsm, args.max_per_problem_math, args.math_cap)
    print(f"  omi gsm={len(omi_gsm)} math={len(omi_math)}", flush=True)

    gsm_orig = load_gsm_train()
    print(f"  gsm8k original train={len(gsm_orig)}", flush=True)

    data = omi_gsm + omi_math + gsm_orig * args.gsm_repeat

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
    )
    before = len(data)
    data = token_len_ok(data, tok, args.max_len)
    print(f"  length filter: {before} -> {len(data)}", flush=True)

    rng = random.Random(0)
    rng.shuffle(data)
    data = data[: args.total_cap]
    print(f"  final {len(data)}", flush=True)

    with open(args.out, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
