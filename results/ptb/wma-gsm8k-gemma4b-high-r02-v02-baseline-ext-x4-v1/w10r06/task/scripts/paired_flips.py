#!/usr/bin/env python3
"""Paired per-item comparison of two inspect-ai gsm8k logs over the same items."""
from __future__ import annotations

import argparse
import json


def load(p):
    d = json.load(open(p))
    return {s["id"]: s["scores"]["match"]["value"] == "C" for s in d["samples"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    ids = sorted(set(A) & set(B))
    gain = [i for i in ids if not A[i] and B[i]]
    loss = [i for i in ids if A[i] and not B[i]]
    res = {"a": args.a, "b": args.b, "n_paired": len(ids),
           "acc_a": sum(A[i] for i in ids) / len(ids),
           "acc_b": sum(B[i] for i in ids) / len(ids),
           "wrong_to_right": len(gain), "right_to_wrong": len(loss),
           "net": len(gain) - len(loss),
           "gain_ids": gain[:40], "loss_ids": loss[:40]}
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
