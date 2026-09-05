#!/usr/bin/env python3
"""Read an inspect-ai gsm8k log and report the diagnostics every card needs:
accuracy, format compliance, stop-reason mix, output length, and the failure list."""
import argparse
import collections
import json
import re

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*\.?\s*$")

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--dump-failures", default=None)
ap.add_argument("--n-show", type=int, default=0)
a = ap.parse_args()

d = json.load(open(a.log))
ss = d["samples"]
sr = collections.Counter()
ends = 0
fails = []
lens = []
for s in ss:
    ch = s["output"]["choices"][0]
    txt = ch["message"]["content"].strip()
    sr[ch.get("stop_reason")] += 1
    lens.append(len(txt))
    if ANS_RE.search(txt):
        ends += 1
    score = list(s["scores"].values())[0]["value"]
    if score != "C":
        fails.append(
            {
                "id": s["id"],
                "question": s["input"] if isinstance(s["input"], str) else str(s["input"])[:400],
                "gold": s["target"],
                "answer_read": list(s["scores"].values())[0].get("answer"),
                "completion": txt,
            }
        )
lens.sort()
print(json.dumps({
    "n": len(ss),
    "accuracy": d["results"]["scores"][0]["metrics"]["accuracy"]["value"],
    "stderr": d["results"]["scores"][0]["metrics"]["stderr"]["value"],
    "ends_with_answer_marker": round(ends / len(ss), 4),
    "stop_reason": dict(sr),
    "chars_p50": lens[len(lens) // 2],
    "chars_p95": lens[int(0.95 * len(lens))],
    "n_failures": len(fails),
}, indent=2))
if a.dump_failures:
    with open(a.dump_failures, "w") as f:
        for x in fails:
            f.write(json.dumps(x) + "\n")
    print("wrote", a.dump_failures)
for x in fails[: a.n_show]:
    print("=" * 80)
    print("GOLD", x["gold"], "| READ", x["answer_read"])
    print(x["completion"][-800:])
