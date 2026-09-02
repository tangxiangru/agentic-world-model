#!/usr/bin/env python3
"""Format-compliance diagnostic over an inspect-ai gsm8k log (same rule as exp-01)."""
import json
import re
import sys

pat = re.compile(r"ANSWER:\s*\$?-?[\d,]+(\.\d+)?\s*$")

log, out = sys.argv[1], sys.argv[2]
d = json.load(open(log))
s = d["samples"]
ok = 0
stops = {}
wrong = []
for x in s:
    c = x["output"]["choices"][0]
    txt = c["message"]["content"].strip()
    if pat.search(txt):
        ok += 1
    stops[c["stop_reason"]] = stops.get(c["stop_reason"], 0) + 1
    if x["scores"]["match"]["value"] != "C":
        wrong.append({"id": x["id"], "target": x["target"],
                      "answer": x["scores"]["match"]["answer"],
                      "tail": txt[-300:]})
res = {"n": len(s), "wellformed_final_answer_line": ok,
       "format_rate": ok / len(s), "stop_reasons": stops,
       "n_wrong": len(wrong), "log": log, "wrong": wrong}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in
                  ("n", "wellformed_final_answer_line", "format_rate",
                   "stop_reasons", "n_wrong")}))
