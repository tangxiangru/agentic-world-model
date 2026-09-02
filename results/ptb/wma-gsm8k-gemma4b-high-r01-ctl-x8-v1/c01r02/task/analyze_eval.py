#!/usr/bin/env python3
"""Diagnostics for the most recent inspect log: the two counters exp-01 measured
(no 'ANSWER:' line, stopped by the token cap) plus the watch-set delta.

Writes analysis/<tag>_failure_tags.json holding sample ids and booleans only -
no question or answer text is copied out of the benchmark (rule 7).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--log", default=None)
ap.add_argument("--watch", default="/home/ben/task/analysis/exp-01_watch.jsonl")
a = ap.parse_args()

log = a.log or max(glob.glob("/home/ben/task/logs/*_gsm8k_*.json"), key=os.path.getmtime)
d = json.load(open(log))
ss = d["samples"]

rec = []
for s in ss:
    ch = s["output"]["choices"][0]
    out = ch["message"]["content"]
    rec.append({
        "id": s["id"],
        "correct": s["scores"]["match"]["value"] == "C",
        "has_answer_line": bool(re.search(r"(?m)^ANSWER:", out)),
        "stop_reason": ch["stop_reason"],
        "n_chars": len(out),
    })

n = len(rec)
summary = {
    "tag": a.tag,
    "log": log,
    "n": n,
    "accuracy": round(sum(r["correct"] for r in rec) / n, 4),
    "no_answer_line": round(sum(not r["has_answer_line"] for r in rec) / n, 4),
    "hit_max_tokens": round(sum(r["stop_reason"] == "max_tokens" for r in rec) / n, 4),
    "mean_chars": round(sum(r["n_chars"] for r in rec) / n, 1),
}

if os.path.exists(a.watch):
    watch = {json.loads(l)["id"] for l in open(a.watch)}
    inrun = {r["id"] for r in rec}
    w = watch & inrun
    summary["watch_set"] = {
        "n_in_this_run": len(w),
        "fixed": sum(1 for r in rec if r["id"] in w and r["correct"]),
        "still_failing": sum(1 for r in rec if r["id"] in w and not r["correct"]),
        "regressions": sum(1 for r in rec if r["id"] not in w and not r["correct"]),
    }

out_path = f"/home/ben/task/analysis/{a.tag}_failure_tags.json"
json.dump({**summary, "items": rec}, open(out_path, "w"), indent=1)
print(json.dumps(summary, indent=2))

fail_path = f"/home/ben/task/analysis/{a.tag}_watch.jsonl"
with open(fail_path, "w") as f:
    for r in rec:
        if not r["correct"]:
            f.write(json.dumps({"id": r["id"]}) + "\n")
print(f"wrote {out_path} and {fail_path}")
