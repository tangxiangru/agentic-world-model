#!/usr/bin/env python3
"""Paired comparison of two inspect eval logs over the same samples (McNemar)."""
from __future__ import annotations

import glob
import json
import sys


def load(path):
    d = json.load(open(path))
    model = d["eval"]["model"]
    out = {}
    for s in d.get("samples", []):
        sc = s.get("scores", {}).get("match", {})
        out[s["id"]] = 1 if sc.get("value") == "C" else 0
    return model, out


def main():
    logs = sorted(glob.glob("logs/*_gsm8k_*.json"))
    runs = []
    for p in logs:
        try:
            m, r = load(p)
        except Exception:
            continue
        if r:
            runs.append((p, m, r))
    for p, m, r in runs:
        print(f"{len(r):4d} {sum(r.values())/len(r):.4f}  {m:32s} {p}")

    if len(sys.argv) == 3:
        a = next(r for r in runs if sys.argv[1] in r[1])
        b = next(r for r in runs if sys.argv[2] in r[1])
        common = set(a[2]) & set(b[2])
        n01 = sum(1 for k in common if a[2][k] == 0 and b[2][k] == 1)
        n10 = sum(1 for k in common if a[2][k] == 1 and b[2][k] == 0)
        print(f"\npaired over n={len(common)}")
        print(f"  {sys.argv[1]}: {sum(a[2][k] for k in common)/len(common):.4f}")
        print(f"  {sys.argv[2]}: {sum(b[2][k] for k in common)/len(common):.4f}")
        print(f"  B-only-correct={n01}  A-only-correct={n10}  diff={(n01-n10)/len(common):+.4f}")
        se = ((n01 + n10) ** 0.5) / len(common)
        print(f"  paired se={se:.4f}  z={(n01-n10)/max((n01+n10)**0.5,1e-9):+.2f}")


if __name__ == "__main__":
    main()
