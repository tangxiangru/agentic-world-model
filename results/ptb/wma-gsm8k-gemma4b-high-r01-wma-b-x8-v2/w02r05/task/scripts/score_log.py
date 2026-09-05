#!/usr/bin/env python3
"""Read the newest inspect json log and report accuracy plus the exp-01
format diagnostic (share of completions ending in an 'ANSWER: <x>' line)."""
from __future__ import annotations

import argparse
import glob
import json
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    log = args.log or sorted(glob.glob("logs/*gsm8k*.json"))[-1]
    d = json.load(open(log))
    s = d["samples"]
    correct = sum(1 for x in s if x["scores"]["match"]["value"] == "C")
    ends = 0
    lens = []
    for x in s:
        c = x["output"]["choices"][0]["message"]["content"].strip()
        if re.search(r"ANSWER:\s*\S+\s*$", c):
            ends += 1
        lens.append(x["output"]["usage"].get("output_tokens", 0))
    rep = {
        "tag": args.tag,
        "log": log,
        "n": len(s),
        "accuracy": round(correct / len(s), 5),
        "ends_with_answer_line": ends,
        "format_share": round(ends / len(s), 4),
        "mean_output_tokens": round(sum(lens) / len(lens), 1),
        "max_output_tokens": max(lens),
        "wrong_ids": [x["id"] for x in s if x["scores"]["match"]["value"] != "C"],
    }
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    r = dict(rep)
    r.pop("wrong_ids")
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
