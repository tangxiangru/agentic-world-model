#!/usr/bin/env python3
"""Summarise an inspect-ai gsm8k eval log: accuracy, termination, format."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="", help="inspect json log; default = newest in logs/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = args.log
    if not path:
        cands = [p for p in glob.glob("logs/*.json") if "gsm8k" in p]
        path = max(cands, key=os.path.getmtime)
    d = json.load(open(path))
    s = d["samples"]
    n = len(s)
    stop_ok = fmt = first_ok = 0
    lens = []
    for x in s:
        ch = x["output"]["choices"][0]
        txt = ch["message"]["content"]
        lens.append(len(txt))
        if ch.get("stop_reason") == "stop":
            stop_ok += 1
        m = re.search(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", txt)
        if m:
            fmt += 1
            try:
                if abs(float(m.group(1).replace(",", "")) - float(x["target"])) < 1e-6:
                    first_ok += 1
            except ValueError:
                pass
    out = {
        "log": path,
        "model": d["eval"]["model"],
        "n": n,
        "accuracy": d["results"]["scores"][0]["metrics"]["accuracy"]["value"],
        "stderr": d["results"]["scores"][0]["metrics"]["stderr"]["value"],
        "stop_reason_stop": stop_ok / n,
        "has_ANSWER_line": fmt / n,
        "first_ANSWER_correct": first_ok / n,
        "mean_chars": sum(lens) / n,
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
