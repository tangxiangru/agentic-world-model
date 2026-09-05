#!/usr/bin/env python3
"""Mix OpenMathInstruct-2 (GSM8K-derived) with on-policy rejection-sampled data."""
from __future__ import annotations

import argparse
import glob
import json
import random

from prep_data import PROMPT, clean_solution, is_plain_number, norm_answer


def load_extra_omi2(need, exclude_keys):
    """Pull additional unique gsm8k-derived rows from the train_5M shards."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
        "snapshots/*/data/train_5M-*.parquet"))
    out = []
    for f in files:
        d = pq.read_table(
            f, columns=["problem", "generated_solution", "expected_answer", "problem_source"]
        ).to_pydict()
        for p, s, a, src in zip(d["problem"], d["generated_solution"],
                                d["expected_answer"], d["problem_source"]):
            if src not in ("gsm8k", "augmented_gsm8k"):
                continue
            a = norm_answer(a)
            if not is_plain_number(a):
                continue
            sc = clean_solution(s)
            if len(sc) < 20:
                continue
            key = (p.strip()[:200], sc[:200])
            if key in exclude_keys:
                continue
            exclude_keys.add(key)
            out.append({
                "prompt": PROMPT.format(prompt=p.strip()),
                "completion": f"{sc}\n\nANSWER: {a}",
                "source": "omi2_5M_" + src,
                "answer": a,
                "question": p.strip(),
            })
            if len(out) >= need:
                return out
        print(f"  scanned {f.split('/')[-1]}: have {len(out)}/{need}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omi2", default="data/sft_gsm.jsonl")
    ap.add_argument("--rft", nargs="*", default=["data/rft1.jsonl"])
    ap.add_argument("--stats", nargs="*", default=["data/rft1_stats.jsonl"])
    ap.add_argument("--n-omi2", type=int, default=155000)
    ap.add_argument("--rft-repeat", type=int, default=2)
    ap.add_argument("--hard-bonus", type=int, default=1,
                    help="extra copies of solutions to problems with low pass rate")
    ap.add_argument("--hard-threshold", type=int, default=6)
    ap.add_argument("--out", default="data/mix2.jsonl")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    omi2 = [json.loads(l) for l in open(args.omi2)]
    print("omi2 pool", len(omi2))
    keys = {(r["question"][:200], r["completion"][:200]) for r in omi2}
    if args.n_omi2 > len(omi2):
        extra = load_extra_omi2(args.n_omi2 - len(omi2), keys)
        print("extra omi2 from train_5M:", len(extra))
        omi2 = omi2 + extra
    rng.shuffle(omi2)
    omi2 = omi2[: args.n_omi2]

    pass_rate = {}
    for sp in args.stats:
        for l in open(sp):
            d = json.loads(l)
            pass_rate[d["question"]] = d["correct"]

    rft = []
    for path in args.rft:
        for l in open(path):
            d = json.loads(l)
            reps = args.rft_repeat
            if pass_rate.get(d["question"], 99) <= args.hard_threshold:
                reps += args.hard_bonus
            rft.extend([d] * reps)
    print("rft examples (after weighting)", len(rft))

    allrows = omi2 + rft
    rng.shuffle(allrows)
    with open(args.out, "w") as f:
        for r in allrows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}: {len(allrows)} rows")


if __name__ == "__main__":
    main()
