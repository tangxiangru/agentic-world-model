#!/usr/bin/env python3
"""Read an inspect-ai gsm8k log and report the numbers the cards need:
graded accuracy, first-ANSWER accuracy, clean-termination rate, stop reasons.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re

from inspect_ai.scorer._common import match_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="inspect json log; default = newest")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = args.log or sorted(glob.glob("logs/*_gsm8k_*.json"))[-1]
    d = json.load(open(path))
    ss = d["samples"]
    rows, first_ok, term = [], 0, 0
    for s in ss:
        c = s["output"]["choices"][0]["message"]["content"]
        m = re.search(r"ANSWER:\s*([^\n]*)", c)
        ok = bool(m) and match_str(m.group(1), s["target"], location="end", numeric=True)[1]
        terminated = bool(m) and len(c[m.end():].strip()) < 3
        first_ok += ok
        term += terminated
        rows.append({"id": s["id"], "target": s["target"],
                     "graded": s["scores"]["match"]["value"],
                     "first_answer_correct": ok, "terminated": terminated,
                     "stop_reason": s["output"]["choices"][0].get("stop_reason")})
    out = {
        "log": path,
        "n": len(ss),
        "graded_accuracy": sum(1 for r in rows if r["graded"] == "C") / len(rows),
        "first_answer_accuracy": first_ok / len(ss),
        "clean_termination_rate": term / len(ss),
        "stop_reason": dict(collections.Counter(r["stop_reason"] for r in rows)),
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
