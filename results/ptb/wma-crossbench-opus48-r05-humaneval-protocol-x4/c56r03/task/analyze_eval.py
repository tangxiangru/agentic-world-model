#!/usr/bin/env python3
"""Summarize an inspect-ai humaneval log: accuracy, fenced-code rate, stop rate,
output-length stats. Usage: python analyze_eval.py <log.json>"""
import json, re, sys

def main(path):
    d = json.load(open(path))
    samples = d.get("samples", [])
    n = len(samples)
    fence = stop = correct = 0
    lens = []
    for s in samples:
        comp = s.get("output", {}).get("completion", "") or ""
        lens.append(len(comp))
        if re.search(r"```(python)?\n", comp):
            fence += 1
        # stop reason
        try:
            sr = s["output"]["stop_reason"]
        except Exception:
            sr = None
        if sr in ("stop", "end_turn", "eos"):
            stop += 1
        sc = s.get("scores", {})
        val = None
        for k, v in sc.items():
            val = v.get("value") if isinstance(v, dict) else v
        if val == "C" or val == "CORRECT" or val == 1:
            correct += 1
    lens.sort()
    print(f"file: {path}")
    print(f"n={n}  accuracy={correct/n:.3f}  fence_rate={fence/n:.3f}  stop_rate={stop/n:.3f}")
    if lens:
        print(f"completion_chars p50={lens[n//2]} p95={lens[int(n*0.95)]} max={lens[-1]}")

if __name__ == "__main__":
    main(sys.argv[1])
