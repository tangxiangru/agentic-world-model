#!/usr/bin/env python3
"""Summarise an inspect-ai gsm8k eval log: accuracy, stop-token share, format share."""
import glob
import json
import re
import sys
from collections import Counter

log = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("logs/*_gsm8k_*.json"))[-1]
out_path = sys.argv[2] if len(sys.argv) > 2 else None
d = json.load(open(log))
S = d["samples"]
n = len(S)
correct = no_fmt = stopped = would_be = 0
gen_lens = []
wrong_ids = []
for s in S:
    ch = s["output"]["choices"][0]
    text = ch["message"]["content"]
    ok = s["scores"]["match"]["value"] == "C"
    correct += ok
    stopped += ch.get("stop_reason") == "stop"
    gen_lens.append(len(text))
    m = re.search(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if not m:
        no_fmt += 1
    else:
        v = m.group(1).replace(",", "").rstrip(".")
        t = s["target"] if isinstance(s["target"], str) else s["target"][0]
        try:
            would_be += abs(float(v) - float(t)) < 1e-6
        except ValueError:
            pass
    if not ok:
        wrong_ids.append(s["id"])
gen_lens.sort()
res = {
    "log": log,
    "n": n,
    "accuracy": correct / n,
    "stopped_share": stopped / n,
    "no_answer_marker_share": no_fmt / n,
    "accuracy_if_truncated_at_first_ANSWER": would_be / n,
    "gen_chars_p50": gen_lens[n // 2],
    "gen_chars_max": gen_lens[-1],
    "stop_reasons": dict(Counter(s["output"]["choices"][0].get("stop_reason") for s in S)),
    "n_wrong": len(wrong_ids),
}
print(json.dumps(res, indent=2))
if out_path:
    res["wrong_ids"] = wrong_ids
    json.dump(res, open(out_path, "w"), indent=2)
