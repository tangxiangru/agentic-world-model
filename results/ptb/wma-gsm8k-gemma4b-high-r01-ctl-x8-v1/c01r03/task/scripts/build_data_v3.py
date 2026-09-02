#!/usr/bin/env python3
"""Build a corpus of rows the exp-04 model has NOT seen, for continued training.

Two kinds of novelty:
  * problems absent from sft_v2 entirely (the extra shards add ~21k)
  * additional, textually different solutions to problems sft_v2 already used
    (OpenMathInstruct-2 carries ~3 solutions per problem across shards, and
    sft_v2 kept only one)

Same renderer and same target shape as build_data.py / build_data_v2.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fmt import ANSWER_MARKER, END_OF_TURN, render_prompt_fast  # noqa: E402
from eval_format import build_system_message, build_user_message  # noqa: E402
from build_data import clean_omi_solution, norm_answer  # noqa: E402

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(TASK_DIR, "data")


def used_pairs(path: str) -> tuple[set[str], set[str]]:
    """(questions, question+solution fingerprints) already trained on."""
    qs, pairs = set(), set()
    for line in open(path):
        e = json.loads(line)
        p = e["prompt"]
        body = p.split("<start_of_turn>user\n", 1)[1].rsplit("<end_of_turn>", 1)[0]
        qs.add(body)
        pairs.add(hash(body) ^ hash(e["completion"]))
    return qs, pairs


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "sft_v3.jsonl"))
    ap.add_argument("--prev", default=os.path.join(DATA, "sft_v2.jsonl"))
    ap.add_argument("--new-problem-cap", type=int, default=21000)
    ap.add_argument("--extra-solutions-cap", type=int, default=45000)
    ap.add_argument("--extra-per-problem", type=int, default=1)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    hold = {
        json.loads(l)["question"]
        for l in open(os.path.join(DATA, "dev_gsm8k_trainholdout.jsonl"))
    }
    prev_prompts, prev_pairs = used_pairs(args.prev)

    paths = sorted(
        glob.glob(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
            "snapshots/*/data/train_*.parquet"
        )
    )
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        frames.append(
            df[df.problem_source.isin(["gsm8k", "augmented_gsm8k"])][
                ["problem", "generated_solution", "expected_answer"]
            ]
        )
    df = pd.concat(frames, ignore_index=True)

    # index of prompts already used, keyed by question text
    prev_q = set()
    for pr in prev_prompts:
        prev_q.add(pr)

    def prompt_of(q: str) -> str:
        return render_prompt_fast(None, build_user_message(q)).split(
            "<start_of_turn>user\n", 1
        )[1].rsplit("<end_of_turn>", 1)[0]

    new_rows, extra_rows = [], []
    per_problem: dict[str, int] = {}
    seen_solution = set()
    for r in df.itertuples(index=False):
        q = str(r.problem).strip()
        if q in hold:
            continue
        ans = norm_answer(r.expected_answer)
        if ans is None:
            continue
        sol = clean_omi_solution(r.generated_solution)
        if not sol or "ANSWER" in sol:
            continue
        target = f"{sol}\n\n{ANSWER_MARKER}{ans}"
        key = hash(q) ^ hash(target)
        if key in seen_solution:
            continue
        seen_solution.add(key)
        pb = prompt_of(q)
        is_new_problem = pb not in prev_q
        if is_new_problem:
            if per_problem.get(q, 0) >= 1:
                continue
            per_problem[q] = per_problem.get(q, 0) + 1
            new_rows.append({"question": q, "target": target})
        else:
            if hash(pb) ^ hash(target + END_OF_TURN) in prev_pairs:
                continue
            if per_problem.get(q, 0) >= args.extra_per_problem:
                continue
            per_problem[q] = per_problem.get(q, 0) + 1
            extra_rows.append({"question": q, "target": target})

    rng.shuffle(new_rows)
    rng.shuffle(extra_rows)
    new_rows = new_rows[: args.new_problem_cap]
    extra_rows = extra_rows[: args.extra_solutions_cap]

    rows = new_rows + extra_rows
    rng.shuffle(rows)
    system = build_system_message()
    n_few = int(len(rows) * args.fewshot_frac)
    out = []
    for i, r in enumerate(rows):
        sysm = system if i < n_few else None
        out.append(
            {
                "prompt": render_prompt_fast(sysm, build_user_message(r["question"])),
                "completion": r["target"].strip() + END_OF_TURN,
                "source": "omi_v3",
                "fewshot": sysm is not None,
            }
        )
    rng.shuffle(out)
    with open(args.out, "w") as f:
        for e in out:
            f.write(json.dumps(e) + "\n")
    print(
        json.dumps(
            {
                "new_problems": len(new_rows),
                "extra_solutions_for_seen_problems": len(extra_rows),
                "total": len(out),
                "fewshot_rows": n_few,
                "out": args.out,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
