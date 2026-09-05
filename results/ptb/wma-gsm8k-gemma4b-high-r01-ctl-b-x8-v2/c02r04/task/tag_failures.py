#!/usr/bin/env python3
"""Diagnostic for a finished eval: how do the wrong answers go wrong?

Reads the newest inspect log (or one named on the command line) and reports the
run-on share that exp-01 measured at 0.59, plus stop reasons and length stats.
No test item text is written out - counts only.
"""
from __future__ import annotations

import glob
import json
import re
import sys

ANS = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")

log = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("logs/*gsm8k*.json"))[-1]
out = sys.argv[2] if len(sys.argv) > 2 else "analysis/failure_tags.json"
d = json.load(open(log))
S = d["samples"]

fail = runon = first_correct = no_marker = 0
stop = {}
lens = []
for s in S:
    ch = s["output"]["choices"][0]
    txt = ch["message"]["content"]
    stop[ch.get("stop_reason")] = stop.get(ch.get("stop_reason"), 0) + 1
    lens.append(len(txt))
    if s["scores"]["match"]["value"] == "C":
        continue
    fail += 1
    m = ANS.search(txt)
    if not m:
        no_marker += 1
        continue
    if len(txt[m.end():].strip()) > 20:
        runon += 1
    try:
        if abs(float(m.group(1).replace(",", "")) - float(s["target"])) < 1e-6:
            first_correct += 1
    except ValueError:
        pass

lens.sort()
rep = {
    "log": log,
    "n": len(S),
    "accuracy": d["results"]["scores"][0]["metrics"]["accuracy"]["value"],
    "failures": fail,
    "failures_runon_past_first_ANSWER": runon,
    "runon_share_of_failures": round(runon / fail, 4) if fail else 0.0,
    "failures_with_no_ANSWER_marker": no_marker,
    "failures_whose_first_ANSWER_was_correct": first_correct,
    "stop_reasons": stop,
    "completion_chars_p50": lens[len(lens) // 2],
    "completion_chars_max": lens[-1],
}
json.dump(rep, open(out, "w"), indent=2)
print(json.dumps(rep, indent=2))
