#!/usr/bin/env python3
"""Summarise the newest inspect-ai gsm8k log: accuracy, stop-token share,
format adherence, and accuracy read at the FIRST 'ANSWER:' line."""
import glob, json, os, re, sys

log = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob("logs/*gsm8k*.json"), key=os.path.getmtime)[-1]
out = sys.argv[2] if len(sys.argv) > 2 else None

d = json.load(open(log))
s = d["samples"]
ok = stopped = fmt = first_ok = 0
lens = []
for x in s:
    c = x["output"]["choices"][0]
    t = c["message"]["content"]
    if isinstance(t, list):
        t = "".join(p.get("text", "") for p in t)
    if x["scores"]["match"]["value"] == "C":
        ok += 1
    if c.get("stop_reason") == "stop":
        stopped += 1
    lines = [l for l in t.strip().splitlines() if l.strip()]
    if lines and re.match(r"^ANSWER:\s*\$?-?[\d,]+", lines[-1].strip()):
        fmt += 1
    m = re.search(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", t)
    if m:
        try:
            if abs(float(m.group(1).replace(",", "")) - float(x["target"])) < 1e-6:
                first_ok += 1
        except ValueError:
            pass
    lens.append(x["output"].get("usage", {}).get("completion_tokens") or 0)

n = len(s)
res = {
    "log": log, "n": n,
    "accuracy": ok / n,
    "stop_share": stopped / n,
    "last_line_is_answer": fmt / n,
    "acc_at_first_answer": first_ok / n,
    "mean_completion_tokens": sum(lens) / max(1, len(lens)),
}
print(json.dumps(res, indent=2))
if out:
    json.dump(res, open(out, "w"), indent=2)
