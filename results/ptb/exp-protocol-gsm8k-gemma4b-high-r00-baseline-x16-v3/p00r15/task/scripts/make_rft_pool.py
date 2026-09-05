#!/usr/bin/env python3
"""Combine rejection-sampled self-solutions with gold solutions for the problems the
model could not solve, into a pool file make_train_set.py can consume.

Rationale: correct self-samples are on-policy and fix the model's own error modes,
but they exist only for problems it already solves, so training on them alone teaches
nothing new about the hard tail. For every problem where all k samples were wrong we
fall back to the OpenMathInstruct-2 gold solution.
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", required=True)
    ap.add_argument("--pool", default="/home/ben/task/data/pool_big.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gold-for-unsolved", type=int, default=2,
                    help="gold solutions to include per problem that RFT never solved")
    ap.add_argument("--gold-for-solved", type=int, default=0,
                    help="gold solutions to also include per solved problem")
    ap.add_argument("--only-sampled-problems", action="store_true",
                    help="restrict the gold fallback to problems that were actually sampled")
    ap.add_argument("--sampled-list", default=None,
                    help="jsonl of {problem: ...} that rft_sample.py attempted")
    args = ap.parse_args()

    rft_rows = [json.loads(l) for l in open(args.rft)]
    solved = {r["problem"] for r in rft_rows}

    sampled = None
    if args.only_sampled_problems and args.sampled_list:
        sampled = {json.loads(l)["problem"] for l in open(args.sampled_list)}

    gold_kept: dict[str, int] = {}
    out_rows = list(rft_rows)
    n_gold_unsolved = n_gold_solved = 0
    for line in open(args.pool):
        r = json.loads(line)
        p = r["problem"]
        if sampled is not None and p not in sampled:
            continue
        cap = args.gold_for_solved if p in solved else args.gold_for_unsolved
        if cap == 0:
            continue
        c = gold_kept.get(p, 0)
        if c >= cap:
            continue
        gold_kept[p] = c + 1
        out_rows.append({k: r[k] for k in ("problem", "prompt", "completion", "answer", "source")})
        if p in solved:
            n_gold_solved += 1
        else:
            n_gold_unsolved += 1

    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"rft rows {len(rft_rows)} over {len(solved)} solved problems; "
          f"gold rows for unsolved {n_gold_unsolved}, gold rows for solved {n_gold_solved}; "
          f"total {len(out_rows)} -> {args.out}")


if __name__ == "__main__":
    main()
