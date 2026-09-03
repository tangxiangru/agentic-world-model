#!/usr/bin/env python3
"""Stage-2 corpus: on-policy rejection-sampled solutions + OpenMathInstruct-2
gsm-type rows that stage 1 (exp-02) never saw.

The "never saw" part is exact, not approximate: load_omi() in build_data.py is
deterministic given (max_per_problem, seed), so re-running it with the same
arguments reproduces the same shuffled row order, and exp-02 consumed
gsm_rows[:95000] / math_rows[:15000]. This takes the rows after those cuts.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import build_data
import fmt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", default="data/rft_exp04.jsonl")
    ap.add_argument("--out", default="data/train_v3.jsonl")
    ap.add_argument("--fresh-gsm", type=int, default=45_000)
    ap.add_argument("--fresh-math", type=int, default=8_000)
    ap.add_argument("--fewshot-rows", type=int, default=1_500)
    ap.add_argument("--stage1-gsm-cut", type=int, default=95_000)
    ap.add_argument("--stage1-math-cut", type=int, default=15_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed + 1)

    # identical call to the one exp-02 used, so the ordering matches
    gsm_all, math_all = build_data.load_omi(2, 10**9, 10**9, 0)
    fresh_gsm = gsm_all[args.stage1_gsm_cut : args.stage1_gsm_cut + args.fresh_gsm]
    fresh_math = math_all[args.stage1_math_cut : args.stage1_math_cut + args.fresh_math]
    print(f"fresh gsm rows {len(fresh_gsm)} (pool {len(gsm_all)}), "
          f"fresh math rows {len(fresh_math)} (pool {len(math_all)})", flush=True)

    fresh = fresh_gsm + fresh_math
    rng.shuffle(fresh)

    sysmsg = fmt.fewshot_system()
    fewshot_pos = set(rng.sample(range(len(fresh)), min(args.fewshot_rows, len(fresh))))

    fresh_rendered = []
    for i, (q, c) in enumerate(fresh):
        system = sysmsg if i in fewshot_pos else None
        p, comp = fmt.render_example(q, c, system=system)
        fresh_rendered.append({"prompt": p, "completion": comp, "fewshot": bool(system)})

    rft_rendered = [json.loads(l) for l in open(args.rft)]
    print(f"rft rows {len(rft_rendered)}", flush=True)

    allrows = fresh_rendered + rft_rendered
    rng.shuffle(allrows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in allrows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(allrows)} rows to {args.out}", flush=True)

    ck = args.out.replace(".jsonl", "_for_contamcheck.jsonl")
    with open(ck, "w") as f:
        for q, c in fresh:
            f.write(json.dumps({"question": q, "answer": c}) + "\n")
    # the rft file already has its own contamcheck dump; append it so one run
    # covers everything the stage-2 trainer reads
    rftck = args.rft.replace(".jsonl", "_for_contamcheck.jsonl")
    if os.path.exists(rftck):
        with open(ck, "a") as f, open(rftck) as g:
            f.write(g.read())
    print("wrote", ck, flush=True)


if __name__ == "__main__":
    main()
