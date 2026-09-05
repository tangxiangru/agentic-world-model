#!/usr/bin/env python3
"""Summarise an inspect-ai gsm8k eval log: accuracy, stop reasons, marker health,
completion length, and the incorrect items (for the next card's watch set)."""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics


def text_of(sample):
    c = sample["output"]["choices"][0]["message"]["content"]
    if isinstance(c, list):
        c = "".join(x.get("text", "") for x in c)
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="eval json; default = newest in logs/")
    ap.add_argument("--dump-wrong", default=None)
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    path = args.log or sorted(glob.glob("logs/*gsm8k*.json"))[-1]
    d = json.load(open(path))
    ss = d["samples"]
    stops = collections.Counter()
    lens, wrong, no_marker, multi_marker = [], [], 0, 0
    n_correct = 0
    for s in ss:
        t = text_of(s)
        stops[s["output"]["choices"][0].get("stop_reason")] += 1
        lens.append(len(t))
        m = t.count("ANSWER:")
        no_marker += m == 0
        multi_marker += m > 1
        sc = list(s["scores"].values())[0]
        if sc["value"] == "C":
            n_correct += 1
        else:
            wrong.append({"id": s["id"], "question": s["input"], "gold": s["target"], "got": sc.get("answer"), "output": t})

    n = len(ss)
    print(f"log {path}")
    print(f"n={n} accuracy={n_correct / n:.4f} correct={n_correct} wrong={len(wrong)}")
    print("stop_reasons", dict(stops))
    print(f"no ANSWER marker={no_marker}  multiple markers={multi_marker}")
    print(f"completion chars p50={statistics.median(lens):.0f} p95={sorted(lens)[int(n * 0.95)]} max={max(lens)}")

    if args.dump_wrong:
        with open(args.dump_wrong, "w") as fh:
            for w in wrong:
                fh.write(json.dumps(w) + "\n")
        print("wrote", args.dump_wrong)
    for w in wrong[: args.show]:
        print("=" * 30, "gold", w["gold"], "got", w["got"])
        print(w["output"][-1200:])


if __name__ == "__main__":
    main()
