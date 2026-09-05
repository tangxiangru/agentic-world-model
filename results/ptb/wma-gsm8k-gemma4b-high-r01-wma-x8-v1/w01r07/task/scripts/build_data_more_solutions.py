#!/usr/bin/env python3
"""Build a second-stage SFT file from *additional* OpenMathInstruct-2 solutions
to the problems already covered.

The full `train` split of OpenMathInstruct-2 contains the same unique problems
as train_1M (verified: 6 of 32 shards contributed exactly 1 unseen gsm8k
problem on top of train_1M's 80818) but many distinct 405B solutions per
problem. Those extra solutions are new supervision for the same questions -
strictly higher quality than self-sampled RFT data, since they come from a much
stronger solver.

Rows already used in a previous stage are excluded by exact completion match.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_data import (  # noqa: E402
    CALC_RE,
    NUM_RE,
    fewshot_block,
    render_completion,
    render_prompt,
    strip_boxed,
)

SNAP = Path(
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
    "469216e3f46f4dacf476b382e192485ea51a143e/data"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="previous stage jsonl files")
    ap.add_argument("--per-problem", type=int, default=2)
    ap.add_argument("--max-rows", type=int, default=95000)
    ap.add_argument("--gsm8k-repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--p-fewshot-short", type=float, default=0.30)
    ap.add_argument("--p-fewshot-full", type=float, default=0.08)
    ap.add_argument("--eval-fewshot-file", default="data/fewshot_system.txt")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    used = set()
    for p in args.exclude:
        with open(p) as f:
            for line in f:
                used.add(json.loads(line)["completion"])
    print(f"[exclude] {len(used)} completions from previous stages")

    per_problem: dict[str, list[str]] = defaultdict(list)
    answers: dict[str, str] = {}
    for shard in sorted(SNAP.glob("train-000*.parquet")):
        tbl = pq.read_table(
            shard,
            columns=["problem", "generated_solution", "expected_answer", "problem_source"],
        ).to_pylist()
        for r in tbl:
            if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                continue
            ans = (r["expected_answer"] or "").strip()
            if not NUM_RE.match(ans):
                continue
            q = r["problem"].strip()
            if len(per_problem[q]) >= args.per_problem * 3:
                continue
            sol = strip_boxed(r["generated_solution"]).strip()
            if not sol or len(sol) > 4000:
                continue
            per_problem[q].append(sol)
            answers[q] = ans
        print(f"  {shard.name}: {len(per_problem)} problems", flush=True)

    eval_fewshot = Path(args.eval_fewshot_file).read_text()

    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    shot_pool = [
        {
            "question": r["question"].strip(),
            "reasoning_raw": "####".join(r["answer"].split("####")[:-1]).strip(),
            "reasoning": CALC_RE.sub("", "####".join(r["answer"].split("####")[:-1]).strip()),
            "answer": r["answer"].split("####")[-1].strip(),
        }
        for r in gsm
    ]

    records = []
    for _ in range(args.gsm8k_repeat):
        for s in shot_pool:
            records.append({"question": s["question"], "reasoning": s["reasoning"], "answer": s["answer"]})

    problems = list(per_problem)
    rng.shuffle(problems)
    for q in problems:
        sols = per_problem[q]
        rng.shuffle(sols)
        taken = 0
        seen_local = set()
        for sol in sols:
            key = re.sub(r"\s+", " ", sol)[:400]
            if key in seen_local:
                continue
            seen_local.add(key)
            records.append({"question": q, "reasoning": sol, "answer": answers[q]})
            taken += 1
            if taken >= args.per_problem:
                break

    rng.shuffle(records)

    n_written = n_dup = 0
    with open(args.out, "w") as f:
        for rec in records:
            completion = render_completion(rec["reasoning"], rec["answer"])
            if completion in used:
                n_dup += 1
                continue
            u = rng.random()
            if u < args.p_fewshot_full:
                prefix = eval_fewshot
            elif u < args.p_fewshot_full + args.p_fewshot_short:
                shots = rng.sample(shot_pool, rng.randint(1, 4))
                prefix = "\n\n".join(
                    fewshot_block(s["question"], s["reasoning_raw"], s["answer"]) for s in shots
                )
            else:
                prefix = None
            f.write(
                json.dumps(
                    {
                        "prompt": render_prompt(rec["question"], prefix),
                        "completion": completion,
                        "src": "openmath2_gsm8k_extra",
                    }
                )
                + "\n"
            )
            n_written += 1
            if n_written >= args.max_rows:
                break
    print(f"wrote {n_written} rows to {args.out} ({n_dup} skipped as already used)")


if __name__ == "__main__":
    main()
