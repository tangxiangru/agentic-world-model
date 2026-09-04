#!/usr/bin/env python3
"""Paired exact McNemar test between two eval item dumps.

Both arms are scored on the same items, so the paired test is far more powerful
than the ~1.8pp unpaired standard error of the difference at n=1319. Written and
self-tested BEFORE exp-03's training launch so case (b) of the pre-committed
decision rule is executed mechanically rather than argued after the number lands.
"""
from __future__ import annotations

import argparse
import json
from math import comb


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = bool(r["correct"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline item dump (jsonl)")
    ap.add_argument("--b", required=True, help="candidate item dump (jsonl)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    ids = sorted(set(A) & set(B))
    if len(ids) != len(A) or len(ids) != len(B):
        print(f"WARNING: {len(A)} vs {len(B)} items, {len(ids)} shared")

    b = sum(1 for i in ids if A[i] and not B[i])       # A right, B wrong
    c = sum(1 for i in ids if B[i] and not A[i])       # B right, A wrong
    n = b + c
    acc_a = sum(A[i] for i in ids) / len(ids)
    acc_b = sum(B[i] for i in ids) / len(ids)

    # two-sided exact binomial on the discordant pairs
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
        p = min(1.0, 2 * tail)

    res = {
        "n_items": len(ids),
        "acc_a": round(acc_a, 4), "acc_b": round(acc_b, 4),
        "delta_b_minus_a": round(acc_b - acc_a, 4),
        "b_only_a_correct": b, "c_only_b_correct": c, "discordant": n,
        "mcnemar_exact_two_sided_p": round(p, 6),
        "label_a": args.label_a, "label_b": args.label_b,
    }
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
