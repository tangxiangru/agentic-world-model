#!/usr/bin/env python3
"""Paired comparison of two inspect eval logs over the identical item set.

All candidates were read greedily on the same 500 items, so the informative
statistic is the discordant pairs (McNemar), not the difference of two
independent-looking accuracies.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from math import comb


def load(log_dir):
    f = sorted(glob.glob(os.path.join(log_dir, "*.json")), key=os.path.getmtime)[-1]
    out = {}
    for s in json.load(open(f)).get("samples") or []:
        sc = list((s.get("scores") or {}).values())
        out[s.get("id")] = bool(sc) and sc[0].get("value") == "C"
    return out


def two_sided_binom_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


a_dir, b_dir = sys.argv[1], sys.argv[2]
A, B = load(a_dir), load(b_dir)
ids = sorted(set(A) & set(B))
b = sum(1 for i in ids if A[i] and not B[i])
c = sum(1 for i in ids if B[i] and not A[i])
print(json.dumps({
    "n_paired": len(ids),
    "acc_a": round(sum(A[i] for i in ids) / len(ids), 4),
    "acc_b": round(sum(B[i] for i in ids) / len(ids), 4),
    "a_only_correct": b,
    "b_only_correct": c,
    "mcnemar_p": round(two_sided_binom_p(b, c), 4),
}, indent=2))
