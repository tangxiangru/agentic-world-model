#!/usr/bin/env python3
"""Summarise an inspect-ai gsm8k log: accuracy, format compliance, length."""
import argparse
import json
import re
import statistics
import sys

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--out", default=None)
a = ap.parse_args()

d = json.load(open(a.log))
s = d["samples"]
acc = d["results"]["scores"][0]["metrics"]["accuracy"]["value"]

ends_answer = 0
cap = 0
lens = []
multi_marker = 0
for x in s:
    t = x["output"]["choices"][0]["message"]["content"]
    lens.append(len(t))
    if re.search(r"ANSWER:\s*\$?-?[\d,]+(\.\d+)?\s*$", t.strip()):
        ends_answer += 1
    if x["output"]["choices"][0]["stop_reason"] == "max_tokens":
        cap += 1
    if len(re.findall(r"ANSWER:", t)) > 1:
        multi_marker += 1

res = {
    "log": a.log,
    "n": len(s),
    "accuracy": acc,
    "ends_with_answer_line": ends_answer / len(s),
    "hit_token_cap": cap / len(s),
    "multiple_answer_markers": multi_marker / len(s),
    "mean_completion_chars": statistics.mean(lens),
}
print(json.dumps(res, indent=2))
if a.out:
    json.dump(res, open(a.out, "w"), indent=2)
