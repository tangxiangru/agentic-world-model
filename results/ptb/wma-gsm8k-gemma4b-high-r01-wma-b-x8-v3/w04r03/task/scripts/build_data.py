#!/usr/bin/env python3
"""Build the SFT corpus for GSM8K from OpenMathInstruct-2 (+ optional gsm8k train).

Targets are shaped for the grader used by evaluate.py:
  * inspect_evals/gsm8k scores with match(numeric=True, location="end"), i.e. the
    LAST number in the completion must be the gold answer,
  * the prompt asks for a final line "ANSWER: $ANSWER",
so every target ends with exactly one "ANSWER: <number>" line and nothing after it.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

import pyarrow.parquet as pq

OMI_DIR = (
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
    "469216e3f46f4dacf476b382e192485ea51a143e/data"
)

NUM_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def unbox(text: str) -> str:
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
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
            k += 1
        out.append(text[j + len("\\boxed{") : k - 1])
        i = k
    return "".join(out)


def clean_solution(sol: str, ans: str) -> str | None:
    s = unbox(sol).strip()
    # collapse the display-math wrappers that only ever held the boxed answer
    s = s.replace("\\[", "").replace("\\]", "")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    if not s:
        return None
    if "ANSWER:" in s.upper():
        return None
    return s + "\n\nANSWER: " + ans


def norm_answer(a: str) -> str | None:
    a = a.strip().replace("$", "").replace("\\!", "").replace("\\%", "").replace("%", "")
    a = a.replace(",", "")
    if not NUM_RE.match(a):
        return None
    if a.startswith("-"):
        return None
    # drop trailing ".0"
    if "." in a:
        a = a.rstrip("0").rstrip(".")
        if a == "":
            return None
    return a


def norm_problem(p: str) -> str:
    return re.sub(r"\s+", " ", p.strip().lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=10)
    ap.add_argument("--n-gsm", type=int, default=150000)
    ap.add_argument("--n-math", type=int, default=12000)
    ap.add_argument("--per-problem", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gsm_by_problem: dict[str, list[dict]] = defaultdict(list)
    math_rows: list[dict] = []

    for i in range(args.shards):
        path = os.path.join(OMI_DIR, f"train-{i:05d}-of-00032.parquet")
        if not os.path.exists(path):
            continue
        tb = pq.read_table(path, columns=["problem", "generated_solution", "expected_answer", "problem_source"])
        cols = tb.to_pydict()
        for prob, sol, ans, src in zip(
            cols["problem"], cols["generated_solution"], cols["expected_answer"], cols["problem_source"]
        ):
            a = norm_answer(ans)
            if a is None:
                continue
            if len(sol) > args.max_chars or len(prob) > 2000:
                continue
            tgt = clean_solution(sol, a)
            if tgt is None:
                continue
            if "\\boxed" in tgt or "\\begin{" in tgt:
                continue
            row = {"question": prob.strip(), "target": tgt, "source": src}
            if src in ("gsm8k", "augmented_gsm8k"):
                key = norm_problem(prob)
                if len(gsm_by_problem[key]) < args.per_problem:
                    gsm_by_problem[key].append(row)
            else:
                if len(math_rows) < args.n_math * 4:
                    math_rows.append(row)
        print(f"shard {i}: gsm problems={len(gsm_by_problem)} math_pool={len(math_rows)}", flush=True)

    gsm_rows = [r for rows in gsm_by_problem.values() for r in rows]
    rng.shuffle(gsm_rows)
    rng.shuffle(math_rows)
    gsm_rows = gsm_rows[: args.n_gsm]
    math_rows = math_rows[: args.n_math]
    rows = gsm_rows + math_rows
    rng.shuffle(rows)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} (gsm={len(gsm_rows)} math={len(math_rows)})")


if __name__ == "__main__":
    main()
