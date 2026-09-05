#!/usr/bin/env python3
"""Diagnostics over an inspect_ai gsm8k JSON log.

--json-output-file only writes {accuracy, stderr}; the mechanism questions this
session asks (does the model terminate? does the grader read the ANSWER line?)
need the per-sample log.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

ANSWER_LINE = re.compile(r"ANSWER:\s*(-?[\d,]+(?:\.\d+)?)\s*$")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def last_number(text: str):
    words = text.strip().split()
    for w in reversed(words):
        w2 = w.strip(".,;:!?$%()[]{}\"'").replace(",", "")
        m = re.fullmatch(r"-?\d+(?:\.\d+)?", w2)
        if m:
            return w2
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = args.log
    if path is None:
        path = max(glob.glob("logs/*_gsm8k_*.json"), key=os.path.getmtime)
    d = json.load(open(path))
    samples = d["samples"]

    n = len(samples)
    n_correct = n_maxtok = n_wellformed = n_ans_marker = 0
    lens = []
    wrong = []
    for s in samples:
        ch = s["output"]["choices"][0]
        out = ch["message"]["content"]
        lens.append(len(out))
        if ch.get("stop_reason") == "max_tokens":
            n_maxtok += 1
        correct = s["scores"]["match"]["value"] == "C"
        n_correct += correct
        cnt = out.count("ANSWER:")
        if cnt >= 1:
            n_ans_marker += 1
        m = ANSWER_LINE.search(out.rstrip())
        if m and last_number(out) == m.group(1).replace(",", ""):
            n_wellformed += 1
        if not correct:
            wrong.append(
                {
                    "id": s["id"],
                    "target": s["target"],
                    "stop_reason": ch.get("stop_reason"),
                    "n_answer_markers": cnt,
                    "last_number": last_number(out),
                    "chars": len(out),
                    "tail": out[-300:],
                }
            )

    lens.sort()
    res = {
        "log": path,
        "n": n,
        "accuracy": n_correct / n,
        "max_tokens_share": n_maxtok / n,
        "wellformed_answer_share": n_wellformed / n,
        "has_answer_marker_share": n_ans_marker / n,
        "chars_p50": lens[n // 2],
        "chars_p90": lens[int(n * 0.9)],
        "chars_max": lens[-1],
        "n_wrong": len(wrong),
    }
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": res, "wrong": wrong}, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
