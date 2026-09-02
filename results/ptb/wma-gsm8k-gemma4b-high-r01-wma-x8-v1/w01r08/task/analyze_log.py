#!/usr/bin/env python3
"""Summarise an inspect_ai gsm8k log: accuracy, contract compliance, stop reasons."""
import json, sys, re, glob, collections

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("logs/*gsm8k*.json"))[-1]
d = json.load(open(path))
ss = d["samples"]
stop = collections.Counter()
ends = 0
correct = 0
lens = []
wrong = []
for s in ss:
    ch = s["output"]["choices"][0]
    stop[ch.get("stop_reason")] += 1
    t = ch["message"]["content"].strip()
    lens.append(len(t))
    if re.search(r"ANSWER:\s*\$?-?[\d,]+\.?\d*\s*$", t):
        ends += 1
    sc = list(s["scores"].values())[0]["value"] if s.get("scores") else None
    if sc == "C":
        correct += 1
    else:
        wrong.append({"id": s["id"], "question": s["input"] if isinstance(s["input"], str) else str(s["input"])[:400],
                      "gold": s["target"], "output_tail": t[-500:]})
n = len(ss)
print(json.dumps({"log": path, "n": n, "accuracy": correct / n,
                  "ends_with_answer_line": ends / n, "stop_reasons": dict(stop),
                  "mean_chars": sum(lens) / n}, indent=2))
if len(sys.argv) > 2:
    with open(sys.argv[2], "w") as f:
        for w in wrong:
            f.write(json.dumps(w) + "\n")
    print("wrote", sys.argv[2], len(wrong), "wrong")
