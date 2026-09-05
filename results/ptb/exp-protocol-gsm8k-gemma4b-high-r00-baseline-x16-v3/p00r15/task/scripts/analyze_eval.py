#!/usr/bin/env python3
"""Summarise an inspect-ai gsm8k eval log: accuracy, termination rate, answer-format rate."""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--dump-failures", default=None)
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    log = json.load(open(args.log))
    samples = log["samples"]
    n = len(samples)
    stop = collections.Counter()
    has_answer = 0
    correct = 0
    lens = []
    failures = []
    for s in samples:
        ch = s["output"]["choices"][0]
        c = ch["message"]["content"]
        stop[ch.get("stop_reason")] += 1
        has_answer += ("ANSWER:" in c)
        lens.append(len(c))
        ok = s["scores"]["match"]["value"] == "C"
        correct += ok
        if not ok:
            failures.append({
                "id": s["id"], "gold": s["target"], "got": s["scores"]["match"]["answer"],
                "stop": ch.get("stop_reason"), "question": s["input"][:400],
                "completion": c,
            })
    print(f"file      {args.log}")
    print(f"n         {n}")
    print(f"accuracy  {correct/n:.4f}  ({correct}/{n})")
    print(f"stop      {dict(stop)}  -> terminated {stop.get('stop',0)/n:.3f}")
    print(f"ANSWER:   {has_answer/n:.3f}")
    print(f"len chars p50 {statistics.median(lens):.0f} max {max(lens)}")
    if args.dump_failures:
        with open(args.dump_failures, "w") as f:
            for x in failures:
                f.write(json.dumps(x) + "\n")
        print(f"wrote {len(failures)} failures -> {args.dump_failures}")
    for x in failures[: args.show]:
        print("=" * 70)
        print("GOLD", x["gold"], "GOT", x["got"], "STOP", x["stop"])
        print(x["completion"][:1500])


if __name__ == "__main__":
    main()
