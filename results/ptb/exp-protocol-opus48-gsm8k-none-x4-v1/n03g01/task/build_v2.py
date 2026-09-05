#!/usr/bin/env python3
"""Combine self-generated (RFT) correct solutions with gold GSM8K solutions."""
import json, argparse, random
from collections import defaultdict


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", default="rft_full.jsonl")
    ap.add_argument("--gold", default="train_sft.jsonl")
    ap.add_argument("--out", default="train_v2.jsonl")
    ap.add_argument("--keep_rft", type=int, default=4)
    ap.add_argument("--include_gold", type=int, default=1)
    args = ap.parse_args()

    rft = load(args.rft)
    gold = load(args.gold)

    by_q = defaultdict(list)
    for r in rft:
        by_q[r["question"]].append(r)

    out = []
    n_solved = 0
    for g in gold:
        q = g["question"]
        rlist = by_q.get(q, [])
        if rlist:
            n_solved += 1
        for r in rlist[:args.keep_rft]:
            out.append({"prompt": r["prompt"], "completion": r["completion"]})
        for _ in range(args.include_gold):
            out.append({"prompt": g["prompt"], "completion": g["completion"]})

    random.seed(0)
    random.shuffle(out)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"gold questions={len(gold)} solved_by_rft={n_solved} "
          f"total_examples={len(out)} rft_solutions={sum(min(len(v),args.keep_rft) for v in by_q.values())}")


if __name__ == "__main__":
    main()
