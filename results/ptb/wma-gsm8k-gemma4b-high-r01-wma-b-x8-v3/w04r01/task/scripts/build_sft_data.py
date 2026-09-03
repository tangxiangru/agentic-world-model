"""Build the SFT jsonl: {prompt, completion} rendered exactly as the grader renders.

Sources
  A. openai/gsm8k split=train (7473) - human reference CoT, terse, same style as the
     10-shot prefix the grader puts in front of every test item.
  B. nvidia/OpenMathInstruct-2 train_1M, problem_source in {gsm8k, augmented_gsm8k}
     - Llama-3.1-405B-Instruct solutions on GSM8K *train* problems and LLM-generated
     variations of them. No test item is involved; the contamination checker is run
     over the built file regardless.

Target shape:  <reasoning>\n\nANSWER: <number><end_of_turn>
Exactly one "ANSWER: " marker per target; the grader reads the last number.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

from datasets import load_dataset, load_from_disk  # noqa: E402

CALC = re.compile(r"<<[^>]*>>")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
# a trailing "\[ 42 \]" display block left behind after unboxing
DISPLAY_TAIL = re.compile(r"\n?\s*\\\[\s*[^\[\]]{0,40}?\s*\\\]\s*$")
NUM_OK = re.compile(r"^-?\d+(?:\.\d+)?$")


def clean_number(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if NUM_OK.match(s):
        # integers stay integers; 18.0 -> 18
        if "." in s:
            f = float(s)
            if f.is_integer():
                return str(int(f))
        return s
    return None


def gsm8k_train_rows():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for r in ds:
        body, _, ans = r["answer"].rpartition("####")
        ans = clean_number(ans)
        if ans is None:
            continue
        body = CALC.sub("", body).strip()
        if not body:
            continue
        yield r["question"], body, ans, "gsm8k_train"


def omi2_rows(path: str, keep_sources=("gsm8k", "augmented_gsm8k")):
    ds = load_from_disk(path)
    keep = set(keep_sources)
    for r in ds:
        if r["problem_source"] not in keep:
            continue
        ans = clean_number(r["expected_answer"])
        if ans is None:
            continue
        body = BOXED.sub(r"\1", r["generated_solution"])
        body = DISPLAY_TAIL.sub("", body).strip()
        if not body or "\\boxed" in body:
            continue
        yield r["problem"], body, ans, r["problem_source"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omi2", default="/home/ben/task/data/omi2_1M")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-omi2", type=int, default=60000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows, seen_sol, per_problem = [], set(), {}

    n_gsm = 0
    for q, body, ans, src in gsm8k_train_rows():
        for _ in range(args.gsm8k_repeat):
            rows.append((q, body, ans, src))
        n_gsm += 1

    pool = []
    for q, body, ans, src in omi2_rows(args.omi2):
        if len(body) > args.max_chars:
            continue
        pool.append((q, body, ans, src))
    rng.shuffle(pool)
    n_omi = 0
    for q, body, ans, src in pool:
        if n_omi >= args.n_omi2:
            break
        if per_problem.get(q, 0) >= args.max_per_problem:
            continue
        key = hash((q, body))
        if key in seen_sol:
            continue
        seen_sol.add(key)
        per_problem[q] = per_problem.get(q, 0) + 1
        rows.append((q, body, ans, src))
        n_omi += 1

    rng.shuffle(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for q, body, ans, src in rows:
            rec = {
                "prompt": fmt.render_prompt(q),
                "completion": fmt.render_completion(body, ans),
                "source": src,
                "question": q,
                "answer": ans,
            }
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} "
          f"(gsm8k_train {n_gsm}x{args.gsm8k_repeat}, omi2 {n_omi})")


if __name__ == "__main__":
    main()
