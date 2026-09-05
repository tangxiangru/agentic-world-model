#!/usr/bin/env python3
"""Summarise an inspect-ai eval log: accuracy, stop reasons, output length,
and whether the graded (last) number came from an 'ANSWER:' line."""
from __future__ import annotations

import json
import re
import sys

path = sys.argv[1]
d = json.load(open(path))
samples = d["samples"]
n = len(samples)
correct = sum(1 for s in samples if s["scores"]["match"]["value"] == "C")
stop = {}
lens = []
has_marker = 0
rows = []
for s in samples:
    out = s["output"]
    ch = out["choices"][0]
    sr = ch.get("stop_reason")
    stop[sr] = stop.get(sr, 0) + 1
    txt = ch["message"]["content"]
    if isinstance(txt, list):
        txt = "".join(p.get("text", "") for p in txt)
    lens.append(out.get("usage", {}).get("completion_tokens") or 0)
    if re.search(r"ANSWER:\s*[-$]?[\d,.]+\s*$", txt.strip()):
        has_marker += 1
    rows.append({
        "id": s["id"],
        "correct": s["scores"]["match"]["value"] == "C",
        "target": s["target"],
        "answer": s["scores"]["match"].get("answer"),
        "stop": sr,
        "tokens": lens[-1],
        "tail": txt.strip()[-160:],
    })

lens.sort()
print(f"file {path}")
print(f"n={n} accuracy={correct / n:.4f} stderr={(correct / n * (1 - correct / n) / n) ** .5:.4f}")
print(f"stop_reason={stop}")
print(f"completion tokens p50={lens[n // 2]} p90={lens[int(n * .9)]} max={lens[-1]}")
print(f"ends with an ANSWER: line: {has_marker}/{n}")
if len(sys.argv) > 2:
    with open(sys.argv[2], "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", sys.argv[2])
