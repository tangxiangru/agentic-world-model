#!/usr/bin/env python3
"""Summarise an inspect-ai eval log: accuracy, format compliance, length, and a
jsonl of the failing items (used as a card's watch_set)."""
from __future__ import annotations

import argparse
import glob
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("--log-dir", required=True)
ap.add_argument("--out-fail", default=None)
ap.add_argument("--out-summary", default=None)
ap.add_argument("--show", type=int, default=0)
a = ap.parse_args()

f = sorted(glob.glob(f"{a.log_dir}/*.json"))[-1]
d = json.load(open(f))
samples = d.get("samples") or []

n = len(samples)
correct = 0
no_marker = 0
truncated = 0
lens = []
fails = []
for s in samples:
    tgt = s["target"] if isinstance(s["target"], str) else s["target"][0]
    out = ""
    msgs = s.get("messages") or []
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            c = m.get("content")
            out = c if isinstance(c, str) else "".join(
                x.get("text", "") for x in c if isinstance(x, dict))
            break
    score = list(s.get("scores", {}).values())
    ok = bool(score) and score[0].get("value") in ("C", 1, 1.0, True)
    correct += ok
    if "ANSWER:" not in out:
        no_marker += 1
    sm = s.get("output", {}).get("stop_reason") or ""
    if sm not in ("stop", "", None):
        truncated += 1
    lens.append(len(out))
    if not ok:
        fails.append({"id": s["id"], "question": s["input"] if isinstance(s["input"], str)
                      else str(s["input"]), "gold": tgt,
                      "model_answer": (score[0].get("answer") if score else None),
                      "output_tail": out[-400:]})

summary = {
    "log": f, "n": n, "accuracy": correct / n if n else None,
    "no_answer_marker": no_marker, "no_answer_marker_share": no_marker / n if n else None,
    "non_stop_finish": truncated, "mean_output_chars": sum(lens) / len(lens) if lens else None,
    "n_failures": len(fails),
}
print(json.dumps(summary, indent=2))
if a.out_fail:
    with open(a.out_fail, "w") as fh:
        for r in fails:
            fh.write(json.dumps(r) + "\n")
if a.out_summary:
    json.dump(summary, open(a.out_summary, "w"), indent=2)
for r in fails[: a.show]:
    print("=" * 70)
    print("Q:", r["question"][:300])
    print("GOLD:", r["gold"], "GOT:", r["model_answer"])
    print("TAIL:", r["output_tail"][-300:])
