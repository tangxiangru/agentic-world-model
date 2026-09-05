#!/usr/bin/env python3
"""Summarise the newest inspect gsm8k log: accuracy, format compliance, truncation."""
from __future__ import annotations

import argparse
import glob
import json
import re

ANS = re.compile(r"ANSWER:\s*\$?-?[\d,]+(?:\.\d+)?\s*$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    log = args.log or sorted(glob.glob("/home/ben/task/logs/*gsm8k*.json"))[-1]
    d = json.load(open(log))
    ss = d["samples"]
    n = len(ss)
    correct = sum(1 for s in ss if s["scores"]["match"]["value"] == "C")
    fmt = trunc = 0
    out_toks = []
    for s in ss:
        ch = s["output"]["choices"][0]
        txt = ch["message"]["content"].strip()
        fmt += bool(ANS.search(txt))
        trunc += ch.get("stop_reason") == "max_tokens"
        out_toks.append(s.get("model_usage", {}))
    res = {
        "tag": args.tag,
        "log": log,
        "n": n,
        "accuracy": correct / n,
        "correct": correct,
        "format_compliant": fmt,
        "format_compliance_rate": fmt / n,
        "hit_max_tokens": trunc,
    }
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
