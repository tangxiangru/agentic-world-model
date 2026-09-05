#!/usr/bin/env python3
"""Build the SFT corpus in the grader's own prompt/target format.

Sources (all GSM8K *train* derived or independent; the GSM8K test split is
never read here -- it is only ever passed to ../contamination_check.py):
  openai/gsm8k main/train              7473 human-written terse CoT
  nvidia/OpenMathInstruct-2            gsm8k + augmented_gsm8k rows, verbose CoT
                                       already verified against expected_answer

Output: data/<name>.jsonl with {prompt, target, question, answer, source}.
`prompt` is byte-exact what vLLM will see; `target` ends in <end_of_turn>.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

GSM8K_TRAIN = glob.glob(
    "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"
)
OMI = sorted(
    glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train-*.parquet"
    )
)


def load_gsm8k_train():
    rows = []
    for p in GSM8K_TRAIN:
        t = pq.read_table(p).to_pylist()
        for r in t:
            reasoning, answer = fmt.clean_gsm8k_reasoning(r["answer"])
            rows.append(
                {
                    "question": r["question"].strip(),
                    "reasoning": reasoning,
                    "answer": answer,
                    "source": "gsm8k_train",
                }
            )
    return rows


_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def load_omi(max_per_problem: int, sources=("gsm8k", "augmented_gsm8k")):
    seen = {}
    rows = []
    for p in OMI:
        t = pq.read_table(p, columns=["problem", "generated_solution", "expected_answer", "problem_source"])
        for r in t.to_pylist():
            if r["problem_source"] not in sources:
                continue
            ans = fmt.normalize_number(r["expected_answer"])
            if not _NUM.match(ans):
                continue  # the grader compares numbers; non-numeric targets teach nothing useful
            q = r["problem"].strip()
            n = seen.get(q, 0)
            if n >= max_per_problem:
                continue
            sol = fmt.clean_omi_solution(r["generated_solution"])
            if "ANSWER:" in sol or "\\boxed" in sol:
                continue  # a second answer marker is the double_answer_format pitfall
            if len(sol) < 40 or len(sol) > 4000:
                continue
            seen[q] = n + 1
            rows.append(
                {
                    "question": q,
                    "reasoning": sol,
                    "answer": ans,
                    "source": "omi2_" + r["problem_source"],
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-omi", type=int, default=48000)
    ap.add_argument("--omi-per-problem", type=int, default=2)
    ap.add_argument("--gsm8k-repeat", type=int, default=1)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gsm = load_gsm8k_train()
    omi = load_omi(args.omi_per_problem)
    print(f"gsm8k_train={len(gsm)} omi_gsm8k_pool={len(omi)}", file=sys.stderr)
    rng.shuffle(omi)
    omi = omi[: args.n_omi]

    rows = gsm * args.gsm8k_repeat + omi
    rng.shuffle(rows)

    # few-shot exemplars are drawn from the GSM8K *train* split only
    shot_pool = [(g["question"], g["reasoning"], g["answer"]) for g in gsm]

    out = []
    for r in rows:
        if rng.random() < args.fewshot_frac:
            k = rng.choice([1, 2, 3, 4, 8, 10])
            shots = rng.sample(shot_pool, k)
            shots = [s for s in shots if s[0] != r["question"]]
        else:
            shots = None
        out.append(
            {
                "prompt": fmt.render_prompt(r["question"], shots),
                "target": fmt.build_target(r["reasoning"], r["answer"]),
                "question": r["question"],
                "answer": r["answer"],
                "source": r["source"],
                "n_shots": len(shots) if shots else 0,
            }
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    from collections import Counter

    print(f"wrote {len(out)} rows to {args.out}", file=sys.stderr)
    print(Counter(o["source"] for o in out), file=sys.stderr)


if __name__ == "__main__":
    main()
