"""Second-round corpus: broaden problem coverage (every unique OpenMathInstruct-2
gsm8k-derived problem, up to N solutions each) and mix in on-policy rejection-
sampled solutions produced by scripts/rft_sample.py.

Same two invariants as round one: the target carries 'ANSWER: ' exactly once and
its last whitespace-separated numeric token is the gold answer.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import fmt as F  # noqa: E402
from scripts.build_data import build_gsm8k_train, build_omi, last_number  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omi", default="data/omi2_gsm8k_5M.jsonl")
    ap.add_argument("--gsm8k-pool", default="data/gsm8k_train_pool.jsonl")
    ap.add_argument("--rft", default=None, help="jsonl from rft_sample.py (question/body/answer/src)")
    ap.add_argument("--omi-cap", type=int, default=170000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="data/sft_train_v2.jsonl")
    ap.add_argument("--decon-out", default="data/sft_train_v2_decon_input.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = build_omi(args.omi, args.max_per_problem, args.omi_cap, rng)
    for _ in range(args.gsm8k_repeat):
        rows += list(build_gsm8k_train(args.gsm8k_pool))
    if args.rft and os.path.exists(args.rft):
        rft = [json.loads(l) for l in open(args.rft)]
        rows += rft
        print(f"  + {len(rft)} rft rows")
    rng.shuffle(rows)

    out, n_bad = [], 0
    for r in rows:
        target = F.render_target(r["body"], r["answer"])
        if target.count(F.ANSWER_MARKER) != 1:
            n_bad += 1
            continue
        if last_number(target[: -len(F.STOP_TOKEN)]) != r["answer"].replace(",", ""):
            n_bad += 1
            continue
        out.append({"question": r["question"], "target": target, "answer": r["answer"],
                    "src": r["src"], "fewshot": rng.random() < args.fewshot_frac})
    print(f"kept {len(out)}  dropped-by-invariant {n_bad}")
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    with open(args.decon_out, "w") as f:
        for r in out:
            f.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")
    print(Counter(r["src"] for r in out), "fewshot rows:", sum(r["fewshot"] for r in out))
    print("wrote", args.out, args.decon_out)


if __name__ == "__main__":
    main()
