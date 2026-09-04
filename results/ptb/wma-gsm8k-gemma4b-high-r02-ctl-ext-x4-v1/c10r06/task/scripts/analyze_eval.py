#!/usr/bin/env python3
"""Diagnostics for an inspect-ai gsm8k eval log: accuracy, format compliance, length."""
import glob, json, re, statistics, sys

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("logs/*gsm8k*.json"))[-1]
d = json.load(open(path))
s = d["samples"]
ok = sum(1 for x in s if x["scores"]["match"]["value"] == "C")
ends = 0
trunc = 0
lens = []
wrong = []
for x in s:
    c = x["output"]["choices"][0]["message"]["content"]
    lens.append(len(c))
    if re.search(r"ANSWER:\s*\$?-?[\d,]+\.?\d*\s*$", c.strip()):
        ends += 1
    if x["output"]["choices"][0].get("stop_reason") == "max_tokens":
        trunc += 1
    if x["scores"]["match"]["value"] != "C":
        wrong.append({"id": x["id"], "target": x["target"],
                      "answer": x["scores"]["match"].get("answer"), "tail": c[-300:]})
print(json.dumps({
    "path": path, "n": len(s), "accuracy": round(ok / len(s), 4),
    "ends_with_answer_line": round(ends / len(s), 4),
    "hit_max_tokens": trunc,
    "chars_p50": int(statistics.median(lens)), "chars_max": max(lens),
}, indent=2))
out = path.replace(".json", "_wrong.json")
json.dump(wrong, open(out, "w"), indent=1)
print("wrong items ->", out, len(wrong))
