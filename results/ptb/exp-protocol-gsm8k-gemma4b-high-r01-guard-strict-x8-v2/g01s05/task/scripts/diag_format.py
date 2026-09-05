#!/usr/bin/env python3
"""Read the newest inspect-ai gsm8k log (or one given by path) and report the
format diagnostic: how many completions end in a well-formed 'ANSWER: <number>'
line, how long they are, and the accuracy."""
from __future__ import annotations

import glob
import json
import re
import sys

FINAL = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")


def text_of(msg) -> str:
    c = msg["content"]
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("logs/*gsm8k*.json"))[-1]
    d = json.load(open(path))
    ss = d["samples"]
    ok_fmt = 0
    lens = []
    correct = 0
    bad = []
    for s in ss:
        t = text_of(s["messages"][-1]).strip()
        lens.append(len(t))
        m = FINAL.search(t)
        if m:
            ok_fmt += 1
        else:
            bad.append({"id": s["id"], "target": s["target"], "tail": t[-200:]})
        if s["scores"]["match"]["value"] == "C":
            correct += 1
    lens.sort()
    out = {
        "log": path,
        "n": len(ss),
        "accuracy": correct / len(ss),
        "format_ok": ok_fmt / len(ss),
        "chars_p50": lens[len(lens) // 2],
        "chars_p90": lens[int(0.9 * len(lens))],
        "chars_max": lens[-1],
        "n_bad_format": len(bad),
        "bad_examples": bad[:5],
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
