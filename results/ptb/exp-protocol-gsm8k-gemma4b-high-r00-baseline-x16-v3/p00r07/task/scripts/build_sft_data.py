#!/usr/bin/env python3
"""Build the SFT file for exp-02.

Sources (both derived from GSM8K's TRAIN split only; the test split is never read):
  * nvidia/OpenMathInstruct-2, rows with problem_source in {gsm8k, augmented_gsm8k}
  * openai/gsm8k train split, gold human solutions

One output style, one answer marker: chain of thought in plain prose, then a
final line "ANSWER: <int>". No \\boxed, no "####", no <<calculator>> spans --
the grader's match(numeric=True, location="end") reads the LAST number in the
completion, so exactly one number may sit at the end.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re

import pandas as pd
from datasets import load_dataset

OMI2_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"

STOP_TOKEN = "<end_of_turn>"

BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")
CALC = re.compile(r"<<[^>]*>>")
INT_ANS = re.compile(r"^-?\d+$")
NUMWORD = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str) -> str | None:
    """Reproduce the grader: last whitespace-token that parses as a number."""
    words = re.split(r"\s+", text.strip())
    for w in reversed(words):
        w2 = w.replace("$", "").replace(",", "").replace("%", "")
        w2 = w2.strip(".!?:;()[]{}\"'")
        if w2.replace(".", "").replace("-", "").isnumeric():
            return w2.lstrip("+")
    return None


def clean_solution(sol: str) -> str:
    sol = sol.replace("\r\n", "\n")
    sol = BOXED.sub(r"\1", sol)
    sol = CALC.sub("", sol)
    sol = re.sub(r"\n{3,}", "\n\n", sol)
    return sol.strip()


def finalize(sol: str, answer: str) -> str | None:
    sol = clean_solution(sol)
    if not sol or "####" in sol or "\\boxed" in sol:
        return None
    body = f"{sol}\n\nANSWER: {answer}"
    # self-check with the grader's own rule, on the text the grader will see
    if last_number(body) != answer.lstrip("+"):
        return None
    # The stop token lives in the data, not in the trainer: the file is then
    # self-describing and preflight's stop_token_consistent check can read it.
    return body + STOP_TOKEN


def load_omi2(max_per_problem: int, want: int, rng: random.Random,
              exclude: set[str] | None = None,
              exclude_pairs: set[str] | None = None) -> list[dict]:
    files = sorted(glob.glob(OMI2_GLOB))
    assert files, "OpenMathInstruct-2 shards not found"
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=["problem", "generated_solution", "expected_answer", "problem_source"])
        df = df[df["problem_source"].isin(["gsm8k", "augmented_gsm8k"])]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["expected_answer"].str.match(INT_ANS, na=False)]
    print(f"omi2 gsm8k-derived integer-answer rows: {len(df)}")

    exclude = exclude or set()
    exclude_pairs = exclude_pairs or set()
    by_problem: dict[str, list[dict]] = {}
    order = list(range(len(df)))
    rng.shuffle(order)
    kept = []
    seen_targets: set[str] = set()
    for i in order:
        r = df.iloc[i]
        p = r["problem"].strip()
        if p in exclude:
            continue
        if len(by_problem.get(p, [])) >= max_per_problem:
            continue
        if not (40 <= len(p) <= 1200):
            continue
        t = finalize(r["generated_solution"], r["expected_answer"].strip())
        if t is None or len(t) > 2500 + len(STOP_TOKEN):
            continue
        key = p + "||" + t
        if key in seen_targets or key in exclude_pairs:
            continue
        seen_targets.add(key)
        row = {"question": p, "target": t, "answer": r["expected_answer"].strip(),
               "src": r["problem_source"]}
        by_problem.setdefault(p, []).append(row)
        kept.append(row)
        if len(kept) >= want:
            break
    print(f"omi2 kept: {len(kept)} over {len(by_problem)} distinct problems")
    return kept


def load_gsm8k_train() -> list[dict]:
    d = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for r in d:
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        if not INT_ANS.match(ans):
            continue
        body = r["answer"].split("####")[0]
        t = finalize(body, ans)
        if t is None:
            continue
        out.append({"question": r["question"].strip(), "target": t, "answer": ans, "src": "gsm8k_train_gold"})
    print(f"gsm8k train gold kept: {len(out)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/sft_v1.jsonl")
    ap.add_argument("--n-omi2", type=int, default=72000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--gold-repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default=None,
                    help="jsonl of already-used rows; their questions are skipped")
    ap.add_argument("--exclude-pairs", default=None,
                    help="jsonl of already-used rows; only those exact (question, target) pairs are skipped, so a problem can contribute fresh solutions")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    exclude = set()
    if args.exclude:
        for line in open(args.exclude):
            exclude.add(json.loads(line)["question"].strip())
        print(f"excluding {len(exclude)} already-used problems")
    exclude_pairs = set()
    if args.exclude_pairs:
        for line in open(args.exclude_pairs):
            d = json.loads(line)
            exclude_pairs.add(d["question"].strip() + "||" + d["target"])
        print(f"excluding {len(exclude_pairs)} already-used (problem, solution) pairs")
    rows = load_omi2(args.max_per_problem, args.n_omi2, rng, exclude, exclude_pairs)
    if args.gold_repeat:
        gold = load_gsm8k_train()
        for _ in range(args.gold_repeat):
            rows.extend(gold)
    rng.shuffle(rows)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
