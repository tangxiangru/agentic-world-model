"""Build the stage-2 SFT corpus: OpenMathInstruct-2 gsm8k rows the stage-1 file
did not contain.

Same pipeline and target format as build_sft_data.py; the only difference is that
up to --max-per-problem solutions per problem are admitted (stage 1 took 2) and
every (question, target) pair already present in the stage-1 file is removed, so
this is genuinely unseen supervision rather than a repeat epoch.
"""
from __future__ import annotations

import argparse
import json
import random

from build_sft_data import gsm8k_train_rows, omi_rows  # noqa: F401  (same filters)
from eval_format import fewshot_system_message, user_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="data/sft_v1.jsonl")
    ap.add_argument("--n-rows", type=int, default=70000)
    ap.add_argument("--max-per-problem", type=int, default=4)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--out", default="data/sft_v2.jsonl")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    seen = set()
    with open(args.stage1) as f:
        for line in f:
            r = json.loads(line)
            seen.add(hash((r["messages"][-2]["content"], r["messages"][-1]["content"])))
    print(f"{len(seen)} (question, target) pairs already used in stage 1")

    rng = random.Random(args.seed)
    pool = omi_rows(args.max_per_problem, rng)
    print(f"pool at max_per_problem={args.max_per_problem}: {len(pool)}")

    fresh = []
    for r in pool:
        if hash((user_prompt(r["question"]), r["target"])) in seen:
            continue
        fresh.append(r)
    print(f"fresh rows: {len(fresh)}")
    rng.shuffle(fresh)
    fresh = fresh[: args.n_rows]

    sysmsg = fewshot_system_message()
    n_fs = 0
    with open(args.out, "w") as f:
        for r in fresh:
            use_fs = rng.random() < args.fewshot_frac
            n_fs += use_fs
            msgs = []
            if use_fs:
                msgs.append({"role": "system", "content": sysmsg})
            msgs.append({"role": "user", "content": user_prompt(r["question"])})
            msgs.append({"role": "assistant", "content": r["target"]})
            f.write(json.dumps({"messages": msgs, "src": r["src"], "fewshot": use_fs,
                                "target": r["target"].strip() + "<end_of_turn>",
                                "text": r["question"] + "\n" + r["target"]}) + "\n")
    print(f"wrote {len(fresh)} rows to {args.out} ({n_fs} with the 10-shot system prefix)")


if __name__ == "__main__":
    main()
