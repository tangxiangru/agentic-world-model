#!/usr/bin/env python3
"""Single-stage corpus: everything exp-02 and exp-04 saw, in one shuffled pass.

Same deterministic load_omi() ordering as build_data.py, so the row slices are
the same rows the two-stage pipeline used, plus the on-policy rejection-sampled
rows from exp-04's sampling pass.
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
    ap.add_argument("--out", default="data/train_v4.jsonl")
    ap.add_argument("--gsm", type=int, default=125_000)
    ap.add_argument("--math", type=int, default=6_000)
    ap.add_argument("--fewshot-rows", type=int, default=1_500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed + 2)
    gsm_all, math_all = build_data.load_omi(2, 10**9, 10**9, 0)
    rows = gsm_all[: args.gsm] + math_all[: args.math] + build_data.load_gsm8k_train()
    rng.shuffle(rows)
    print(f"omi+gsm8k rows {len(rows)}", flush=True)

    sysmsg = fmt.fewshot_system()
    fewshot_pos = set(rng.sample(range(len(rows)), min(args.fewshot_rows, len(rows))))
    rendered = []
    for i, (q, c) in enumerate(rows):
        system = sysmsg if i in fewshot_pos else None
        p, comp = fmt.render_example(q, c, system=system)
        rendered.append({"prompt": p, "completion": comp, "fewshot": bool(system)})

    rft = [json.loads(l) for l in open(args.rft)]
    print(f"rft rows {len(rft)}", flush=True)
    allrows = rendered + rft
    rng.shuffle(allrows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in allrows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(allrows)} rows to {args.out}", flush=True)

    ck = args.out.replace(".jsonl", "_for_contamcheck.jsonl")
    with open(ck, "w") as f:
        for q, c in rows:
            f.write(json.dumps({"question": q, "answer": c}) + "\n")
    rftck = args.rft.replace(".jsonl", "_for_contamcheck.jsonl")
    if os.path.exists(rftck):
        with open(ck, "a") as f, open(rftck) as g:
            f.write(g.read())
    print("wrote", ck, flush=True)


if __name__ == "__main__":
    main()
