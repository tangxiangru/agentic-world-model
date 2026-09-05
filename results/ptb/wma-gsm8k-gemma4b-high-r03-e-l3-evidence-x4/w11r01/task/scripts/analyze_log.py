#!/usr/bin/env python3
"""Read an inspect json log: accuracy, stop reasons, format compliance, run-on rate."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter

path = sys.argv[1]
d = json.load(open(path))
samples = d["samples"]
n = len(samples)
correct = sum(1 for s in samples if s["scores"]["match"]["value"] == "C")
stops = Counter()
has_answer = 0
runon = 0
lens = []
for s in samples:
    out = s["output"]
    comp = out["choices"][0]["message"]["content"]
    if isinstance(comp, list):
        comp = "".join(c.get("text", "") for c in comp)
    stops[out["choices"][0].get("stop_reason")] += 1
    lens.append(out.get("usage", {}).get("completion_tokens", 0))
    if re.search(r"(?m)^\s*ANSWER:", comp):
        has_answer += 1
        tail = comp.split("ANSWER:")[-1]
        # anything substantial after the first ANSWER line = the model kept going
        if len(tail.strip().split("\n", 1)) > 1 and len(tail.strip()) > 30:
            runon += 1

lens.sort()
print(f"file={path}")
print(f"n={n} accuracy={correct / n:.4f}")
print(f"stop_reasons={dict(stops)}")
print(f"has ANSWER: line = {has_answer}/{n} ({has_answer / n:.3f})")
print(f"kept going after first ANSWER: = {runon}/{n} ({runon / n:.3f})")
print(f"completion tokens p50={lens[n // 2]} p90={lens[int(n * 0.9)]} max={lens[-1]}")

if len(sys.argv) > 2 and sys.argv[2] == "--show":
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    for s in samples[:k]:
        comp = s["output"]["choices"][0]["message"]["content"]
        if isinstance(comp, list):
            comp = "".join(c.get("text", "") for c in comp)
        print("=" * 70)
        print("TARGET:", s["target"], "| SCORED:", s["scores"]["match"]["answer"],
              "|", s["scores"]["match"]["value"])
        print(comp[:2500])
