#!/usr/bin/env python3
"""Summarise an inspect-ai gsm8k json log: accuracy, stop reasons, answer shape."""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=None)
    ap.add_argument("--dump-wrong", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = args.log or sorted(glob.glob("logs/*_gsm8k_*.json"))[-1]
    d = json.load(open(path))
    ss = d["samples"]
    n = len(ss)
    correct = sum(1 for s in ss if s["scores"]["match"]["value"] == "C")
    stops: dict[str, int] = {}
    ans_line = 0
    lens = []
    wrong = []
    for s in ss:
        ch = s["output"]["choices"][0]
        r = ch.get("stop_reason", "?")
        stops[r] = stops.get(r, 0) + 1
        c = ch["message"]["content"]
        lens.append(len(c))
        if re.search(r"ANSWER:\s*-?[\d,]+(\.\d+)?\s*$", c.strip()):
            ans_line += 1
        if s["scores"]["match"]["value"] != "C":
            wrong.append({"id": s["id"], "target": s["target"],
                          "answer": s["scores"]["match"]["answer"],
                          "input": s["input"] if isinstance(s["input"], str) else "",
                          "completion": c})
    summary = {
        "log": path,
        "n": n,
        "accuracy": correct / n,
        "stop_reasons": stops,
        "share_ends_with_answer_line": ans_line / n,
        "mean_completion_chars": sum(lens) / n,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        json.dump({"summary": summary, "wrong": wrong}, open(args.out, "w"), indent=2)
    for w in wrong[: args.dump_wrong]:
        print("=" * 70)
        print("gold", w["target"], "| read", w["answer"])
        print(w["completion"][-1500:])


if __name__ == "__main__":
    sys.exit(main())
