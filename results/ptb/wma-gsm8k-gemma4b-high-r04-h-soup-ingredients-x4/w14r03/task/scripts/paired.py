#!/usr/bin/env python3
"""Paired (McNemar) comparison of two inspect eval logs on the same items."""
from __future__ import annotations

import argparse
import json
from math import comb


def scores(path):
    d = json.load(open(path))
    out = {}
    for s in d["samples"]:
        sc = s.get("scores") or {}
        out[str(s["id"])] = 1 if any(v.get("value") == "C" for v in sc.values()) else 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline log")
    ap.add_argument("--b", required=True, help="candidate log")
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    a, b = scores(x.a), scores(x.b)
    ids = sorted(set(a) & set(b))
    fixed = sum(1 for i in ids if a[i] == 0 and b[i] == 1)
    reg = sum(1 for i in ids if a[i] == 1 and b[i] == 0)
    n = fixed + reg
    p = min(1.0, sum(comb(n, k) for k in range(min(fixed, reg) + 1)) / 2 ** n * 2) if n else 1.0
    r = {"n_paired": len(ids), "acc_a": sum(a[i] for i in ids) / len(ids),
         "acc_b": sum(b[i] for i in ids) / len(ids),
         "fixed": fixed, "regressed": reg, "net": fixed - reg, "mcnemar_p": round(p, 4),
         "log_a": x.a, "log_b": x.b}
    json.dump(r, open(x.out, "w"), indent=2)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
