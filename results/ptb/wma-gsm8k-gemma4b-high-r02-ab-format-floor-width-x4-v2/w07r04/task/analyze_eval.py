#!/usr/bin/env python3
"""Termination/format diagnostics from an inspect-ai eval log directory."""
import glob
import json
import re
import sys

log_dir, out_path = sys.argv[1], sys.argv[2]
f = sorted(glob.glob(f"{log_dir}/*.json"))[-1]
d = json.load(open(f))
S = d["samples"]
rows = []
for s in S:
    c = s["output"]["choices"][0]["message"]["content"]
    if isinstance(c, list):
        c = "".join(x.get("text", "") for x in c)
    m = re.search(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", c)
    rows.append({
        "id": s["id"],
        "scored": s["scores"]["match"]["value"],
        "has_answer_line": "ANSWER:" in c,
        "n_answer_lines": c.count("ANSWER:"),
        "stop_reason": s["output"]["choices"][0].get("stop_reason"),
        "first_answer": m.group(1) if m else None,
        "gold": s["target"],
        "chars": len(c),
    })
n = len(rows)
summary = {
    "log": f,
    "n": n,
    "accuracy": sum(r["scored"] == "C" for r in rows) / n,
    "no_answer_line": sum(not r["has_answer_line"] for r in rows),
    "multi_answer_lines": sum(r["n_answer_lines"] > 1 for r in rows),
    "hit_max_tokens": sum(r["stop_reason"] == "max_tokens" for r in rows),
    "median_chars": sorted(r["chars"] for r in rows)[n // 2],
    "correct_on_first_answer_line": sum(
        1 for r in rows if r["first_answer"] is not None
        and r["first_answer"].replace(",", "").rstrip(".").split(".")[0] == r["gold"].replace(",", "")),
}
json.dump({**summary, "samples": rows}, open(out_path, "w"), indent=1)
print(json.dumps(summary, indent=1))
