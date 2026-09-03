#!/usr/bin/env python3
"""Build the SFT corpus for GSM8K.

Sources
  * nvidia/OpenMathInstruct-2 (rev 469216e3), problem_source in
    {gsm8k, augmented_gsm8k} -> the bulk; {math, augmented_math} -> a small
    generalisation tail.
  * openai/gsm8k train split (7473 items), human solutions, verbatim reasoning.

Every target is reshaped into exactly one format:

    <step by step reasoning>

    ANSWER: <number>

which is what inspect_evals/gsm8k asks for and what match(numeric=True,
location="end") reads (the LAST numeric token of the completion).

Nothing here touches the GSM8K test split.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq

import fmt

OMI_GLOB = (
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
    "469216e3f46f4dacf476b382e192485ea51a143e/data/train-*.parquet"
)

BOXED_RE = re.compile(r"\\boxed\s*\{")
NUM_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
    out = []
    i = 0
    while True:
        m = BOXED_RE.search(text, i)
        if not m:
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


def clean_number(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUM_RE.match(a):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    return a


def make_completion(reasoning: str, answer: str) -> str | None:
    reasoning = reasoning.strip()
    if not reasoning:
        return None
    # the grader's marker must appear exactly once, and only at the end
    if fmt.ANSWER_MARKER in reasoning or "ANSWER:" in reasoning:
        return None
    return f"{reasoning}\n\n{fmt.ANSWER_MARKER}{answer}"


def load_omi(max_per_problem: int, want_gsm: int, want_math: int, seed: int):
    files = sorted(glob.glob(OMI_GLOB))
    assert files, OMI_GLOB
    rng = random.Random(seed)
    gsm_by_problem: dict[str, list] = defaultdict(list)
    math_rows = []
    n_seen = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=50_000):
            d = batch.to_pydict()
            for prob, sol, ans, src in zip(
                d["problem"], d["generated_solution"], d["expected_answer"], d["problem_source"]
            ):
                n_seen += 1
                is_gsm = src in ("gsm8k", "augmented_gsm8k")
                if is_gsm:
                    if len(gsm_by_problem[prob]) >= max_per_problem:
                        continue
                    a = clean_number(ans)
                    if a is None:
                        continue
                    body = strip_boxed(sol)
                    c = make_completion(body, a)
                    if c is None or len(c) > 4000:
                        continue
                    gsm_by_problem[prob].append(c)
                else:
                    # reservoir-ish: cheap subsample of the math tail
                    if rng.random() > 0.02:
                        continue
                    a = ans.strip()
                    if not a or len(a) > 40:
                        continue
                    body = strip_boxed(sol)
                    c = make_completion(body, a)
                    if c is None or len(c) > 4000:
                        continue
                    math_rows.append((prob, c))
    print(f"scanned {n_seen} OMI rows; gsm problems={len(gsm_by_problem)} "
          f"gsm rows={sum(len(v) for v in gsm_by_problem.values())} math pool={len(math_rows)}")

    gsm_rows = [(p, c) for p, cs in gsm_by_problem.items() for c in cs]
    rng.shuffle(gsm_rows)
    rng.shuffle(math_rows)
    return gsm_rows[:want_gsm], math_rows[:want_math]


def load_gsm8k_train():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for r in ds:
        body, _, ans = r["answer"].rpartition("####")
        a = clean_number(ans)
        if a is None:
            continue
        c = make_completion(body, a)
        if c is None:
            continue
        rows.append((r["question"], c))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--gsm", type=int, default=140_000)
    ap.add_argument("--math", type=int, default=25_000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-rows", type=int, default=8_000,
                    help="rows rendered with the grader's exact 10-shot system prefix")
    ap.add_argument("--gsm8k-train-repeat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gsm_rows, math_rows = load_omi(args.max_per_problem, args.gsm, args.math, args.seed)
    orig = load_gsm8k_train()
    print(f"gsm8k train original rows: {len(orig)}")

    pool = gsm_rows + math_rows + orig * args.gsm8k_train_repeat
    rng.shuffle(pool)

    sysmsg = fmt.fewshot_system()
    # the fewshot-prefixed rows are drawn only from gsm-type rows
    fewshot_idx = set()
    gsm_positions = [i for i, (p, c) in enumerate(pool)]
    rng.shuffle(gsm_positions)
    fewshot_idx = set(gsm_positions[: args.fewshot_rows])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for i, (q, c) in enumerate(pool):
            system = sysmsg if i in fewshot_idx else None
            prompt, completion = fmt.render_example(q, c, system=system)
            f.write(json.dumps({"prompt": prompt, "completion": completion,
                                "fewshot": bool(system)}) + "\n")
            n += 1
    print(f"wrote {n} rows to {args.out}  (fewshot-prefixed: {len(fewshot_idx)})")

    # a plain-text dump for the contamination checker: question + solution only
    ck = args.out.replace(".jsonl", "_for_contamcheck.jsonl")
    with open(ck, "w") as f:
        for q, c in pool:
            f.write(json.dumps({"question": q, "answer": c}) + "\n")
    print("wrote", ck)


if __name__ == "__main__":
    main()
