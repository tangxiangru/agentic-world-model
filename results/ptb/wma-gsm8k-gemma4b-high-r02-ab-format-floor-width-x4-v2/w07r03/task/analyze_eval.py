#!/usr/bin/env python3
"""Read the newest inspect_ai log and print the C11 instrumentation the cards
ask for: score, stop-reason histogram, truncation rate, output-length stats,
missing-marker count and garbage-prefix count."""
import glob, json, os, statistics, sys

f = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob("logs/*gsm8k*.json"), key=os.path.getmtime)[-1]
L = json.load(open(f))
S = L["samples"]
stop, toks, noans, garbage, wrong = {}, [], 0, 0, 0
for s in S:
    c = s["output"]["choices"][0]
    t = c["message"]["content"]
    if isinstance(t, list):
        t = "".join(p.get("text", "") for p in t)
    stop[c.get("stop_reason")] = stop.get(c.get("stop_reason"), 0) + 1
    toks.append(s["output"]["usage"]["output_tokens"])
    if "ANSWER:" not in t:
        noans += 1
    # a well-formed answer starts with prose or a digit, not a stray marker
    if t[:1] in ("<", "|", "�") or t.lstrip()[:12].count("\n") > 2:
        garbage += 1
    if s["scores"]["match"]["value"] != "C":
        wrong += 1
n = len(S)
out = {
    "log": f,
    "n": n,
    "accuracy": round((n - wrong) / n, 4),
    "stop_reasons": stop,
    "truncation_rate": round(stop.get("max_tokens", 0) / n, 4),
    "no_answer_marker": noans,
    "garbage_prefix": garbage,
    "output_tokens": {"mean": round(statistics.mean(toks), 1),
                      "p50": statistics.median(toks), "max": max(toks)},
}
print(json.dumps(out, indent=2))
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=2)
