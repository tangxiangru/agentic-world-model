#!/usr/bin/env python3
"""Paired item-level comparison between inspect_ai eval logs.

Accuracy differences of one or two points at these sample sizes are decided by
the discordant items, not by the marginals, so every pair gets fixed/broken
counts and an exact two-sided McNemar test. Items are matched on samples[].id.
"""
import argparse, itertools, json
from math import comb


def scores(path):
    log = json.load(open(path))
    out = {}
    for s in log["samples"]:
        v = list(s["scores"].values())[0]["value"]
        out[s["id"]] = 1 if v in ("C", 1, 1.0, True) else 0
    return out


def mcnemar(b01, b10):
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    return min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0)


ap = argparse.ArgumentParser()
ap.add_argument("--pairs", nargs="+", required=True, help="name=logpath")
ap.add_argument("--out", default=None)
a = ap.parse_args()

data = {}
for spec in a.pairs:
    name, path = spec.split("=", 1)
    data[name] = scores(path)

report = {"marginals": {}, "pairs": []}
for name, s in data.items():
    report["marginals"][name] = {"n": len(s), "correct": sum(s.values()), "accuracy": sum(s.values()) / len(s)}

for x, y in itertools.combinations(data, 2):
    ids = sorted(set(data[x]) & set(data[y]))
    b01 = sum(1 for i in ids if data[x][i] == 0 and data[y][i] == 1)
    b10 = sum(1 for i in ids if data[x][i] == 1 and data[y][i] == 0)
    report["pairs"].append(
        {
            "a": x, "b": y, "paired_items": len(ids),
            "a_acc": sum(data[x][i] for i in ids) / len(ids),
            "b_acc": sum(data[y][i] for i in ids) / len(ids),
            f"{y}_fixes": b01, f"{y}_breaks": b10,
            "discordant": b01 + b10, "mcnemar_p": mcnemar(b01, b10),
        }
    )

js = json.dumps(report, indent=2)
print(js)
if a.out:
    open(a.out, "w").write(js)
