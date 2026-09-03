#!/usr/bin/env python3
"""Re-filter the saved raw RFT generations without resampling.

exp-05 failed because sample_rft.py sorted correct candidates by length and kept
the two shortest, selecting for terseness. Here the kept chains are drawn at
random from the correct ones, with a floor on chain length so degenerate
one-liners cannot win a slot.
"""
from __future__ import annotations

import argparse
import json
import random
import re

STOP = "<end_of_turn>"
ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")


def norm(x: str):
    x = x.replace(",", "").strip()
    try:
        f = float(x)
    except ValueError:
        return None
    if f != f or abs(f) == float("inf") or abs(f) > 1e15:
        return None
    return str(int(f)) if f == int(f) else str(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/rft_v1_raw.jsonl")
    ap.add_argument("--out", default="data/rft_v2.jsonl")
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows, stats = [], {"solved": 0, "unsolved": 0, "kept": 0, "problems": 0}
    lens = []
    for line in open(a.raw):
        r = json.loads(line)
        stats["problems"] += 1
        gold = norm(r["gold"])
        cands = []
        for t in r["samples"]:
            t = t.strip()
            m = ANS_RE.search(t)
            if not m or norm(m.group(1)) != gold:
                continue
            body = t[: m.start()].rstrip()
            if len(body) < a.min_chars or "ANSWER:" in body:
                continue
            cands.append(f"{body}\n\nANSWER: {gold}{STOP}")
        cands = list(dict.fromkeys(cands))
        if not cands:
            stats["unsolved"] += 1
            continue
        stats["solved"] += 1
        rng.shuffle(cands)
        for c in cands[: a.keep]:
            rows.append({"prompt": r["prompt"], "completion": c,
                         "source": f"rft2_{r['src']}", "question": r["q"]})
            stats["kept"] += 1
            lens.append(len(c))

    rng.shuffle(rows)
    with open(a.out, "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    lens.sort()
    stats["chain_chars_p50"] = lens[len(lens) // 2]
    stats["chain_chars_mean"] = round(sum(lens) / len(lens), 1)
    with open(a.out.replace(".jsonl", "_report.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
