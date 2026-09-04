#!/usr/bin/env python3
"""Paired (McNemar) comparison of two eval dumps scored on the identical slice.

The marginal stderr of each arm is the wrong yardstick when both arms answer the
same items: what decides the ranking is the discordant pairs - items one arm gets
right and the other gets wrong.

  python scripts/paired_compare.py analysis/A.jsonl analysis/B.jsonl
"""
from __future__ import annotations

import argparse
import json
import math


def load(path: str) -> dict:
    return {json.loads(l)["id"]: json.loads(l)["correct"] for l in open(path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--name-a", default=None)
    ap.add_argument("--name-b", default=None)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    na = args.name_a or args.a
    nb = args.name_b or args.b
    ids = sorted(set(A) & set(B))
    if len(ids) != len(A) or len(ids) != len(B):
        print(f"WARNING: {len(A)} vs {len(B)} records, {len(ids)} joinable")

    both = sum(A[i] and B[i] for i in ids)
    only_a = sum(A[i] and not B[i] for i in ids)
    only_b = sum(B[i] and not A[i] for i in ids)
    neither = sum(not A[i] and not B[i] for i in ids)
    n = len(ids)

    acc_a, acc_b = (both + only_a) / n, (both + only_b) / n
    d = only_a - only_b
    disc = only_a + only_b
    # standard error of the PAIRED difference in proportions
    se = math.sqrt(disc) / n if disc else 0.0
    z = d / math.sqrt(disc) if disc else 0.0

    print(f"n joined            : {n}")
    print(f"{na:28s}: {acc_a:.4f}")
    print(f"{nb:28s}: {acc_b:.4f}")
    print(f"both correct        : {both}")
    print(f"only {na:23s}: {only_a}")
    print(f"only {nb:23s}: {only_b}")
    print(f"neither             : {neither}")
    print(f"paired difference   : {acc_a - acc_b:+.4f}  (paired se {se:.4f}, z {z:+.2f})")
    if disc == 0:
        print("verdict             : identical on every item")
    elif abs(z) >= 1.96:
        print(f"verdict             : {na if d > 0 else nb} wins (|z| >= 1.96)")
    elif abs(z) >= 1.0:
        print(f"verdict             : leans {na if d > 0 else nb}, not significant")
    else:
        print("verdict             : unresolved - the discordant pairs are balanced")


if __name__ == "__main__":
    main()
