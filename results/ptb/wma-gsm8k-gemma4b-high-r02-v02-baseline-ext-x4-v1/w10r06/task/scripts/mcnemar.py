#!/usr/bin/env python3
"""Paired comparison of two inspect-ai logs: flips, McNemar exact p, subset check."""
from __future__ import annotations

import argparse
import json
from math import comb


def load(p):
    d = json.load(open(p))
    return {s["id"]: s["scores"]["match"]["value"] == "C" for s in d["samples"]}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--subset", default=None,
                    help="log whose ids define a subset to also report on")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    ids = sorted(set(A) & set(B))
    b = sum(1 for i in ids if A[i] and not B[i])   # A right, B wrong
    c = sum(1 for i in ids if not A[i] and B[i])   # A wrong, B right
    res = {
        "a": {"label": args.label_a, "log": args.a,
              "acc": sum(A[i] for i in ids) / len(ids)},
        "b": {"label": args.label_b, "log": args.b,
              "acc": sum(B[i] for i in ids) / len(ids)},
        "n_paired": len(ids),
        "a_right_b_wrong": b, "a_wrong_b_right": c,
        "net_for_b": c - b,
        "mcnemar_exact_p": mcnemar_exact(b, c),
    }
    if args.subset:
        S = set(load(args.subset))
        sids = [i for i in ids if i in S]
        res["subset"] = {
            "n": len(sids),
            "acc_a": sum(A[i] for i in sids) / len(sids),
            "acc_b": sum(B[i] for i in sids) / len(sids),
        }
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
