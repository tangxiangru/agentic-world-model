#!/usr/bin/env python3
"""Paired per-item comparison of two inspect gsm8k logs scored on the same items.

Aggregate accuracy on n=300 has a ~3 pp floor; the paired count of
A-right/B-wrong vs B-right/A-wrong has a much smaller one, because the two arms
answered the same questions.
"""
from __future__ import annotations

import argparse
import json
import math


def load(path: str) -> dict[str, bool]:
    d = json.load(open(path))
    out = {}
    for s in d.get("samples", []):
        out[str(s.get("id"))] = list(s["scores"].values())[0]["value"] == "C"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    ids = sorted(set(A) & set(B))
    a_only = sum(1 for i in ids if A[i] and not B[i])
    b_only = sum(1 for i in ids if B[i] and not A[i])
    both = sum(1 for i in ids if A[i] and B[i])
    neither = len(ids) - both - a_only - b_only
    n_disc = a_only + b_only
    # exact two-sided sign test on the discordant pairs (McNemar)
    if n_disc:
        k = min(a_only, b_only)
        p = min(1.0, 2 * sum(math.comb(n_disc, i) for i in range(k + 1)) / 2 ** n_disc)
    else:
        p = 1.0
    res = {
        "n_paired": len(ids),
        f"acc_{args.name_a}": (both + a_only) / len(ids),
        f"acc_{args.name_b}": (both + b_only) / len(ids),
        "both_right": both, "both_wrong": neither,
        f"{args.name_a}_only": a_only, f"{args.name_b}_only": b_only,
        "discordant": n_disc,
        "mcnemar_exact_p": round(p, 4),
    }
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
