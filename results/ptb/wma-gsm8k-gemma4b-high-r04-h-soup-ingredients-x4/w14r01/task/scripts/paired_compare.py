#!/usr/bin/env python3
"""McNemar paired comparison of two inspect eval logs over the same items."""
from __future__ import annotations

import argparse
import glob
import json
import os


def load(path_or_dir):
    p = path_or_dir
    if os.path.isdir(p):
        cands = [c for c in glob.glob(os.path.join(p, "*.json")) if os.path.getsize(c) > 10000]
        p = sorted(cands, key=os.path.getmtime)[-1]
    log = json.load(open(p))
    out = {}
    for s in log.get("samples") or []:
        sc = list(s.get("scores", {}).values())
        out[s.get("id")] = bool(sc) and sc[0].get("value") == "C"
    return p, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    pa, a = load(args.a)
    pb, b = load(args.b)
    ids = sorted(set(a) & set(b))
    a_only = sum(1 for i in ids if a[i] and not b[i])
    b_only = sum(1 for i in ids if b[i] and not a[i])
    both = sum(1 for i in ids if a[i] and b[i])
    neither = sum(1 for i in ids if not a[i] and not b[i])
    n = len(ids)
    disc = a_only + b_only
    # exact two-sided binomial p on the discordant pairs
    from math import comb

    k = min(a_only, b_only)
    p = min(1.0, 2 * sum(comb(disc, i) for i in range(k + 1)) / (2 ** disc)) if disc else 1.0
    print(
        json.dumps(
            {
                "a": pa,
                "b": pb,
                "n_common": n,
                "a_acc": round(sum(a[i] for i in ids) / n, 4),
                "b_acc": round(sum(b[i] for i in ids) / n, 4),
                "both_correct": both,
                "neither": neither,
                "a_only": a_only,
                "b_only": b_only,
                "discordant": disc,
                "mcnemar_exact_p": round(p, 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
