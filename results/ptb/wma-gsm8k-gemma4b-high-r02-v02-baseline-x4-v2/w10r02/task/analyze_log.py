#!/usr/bin/env python3
"""Read an inspect-ai json eval log and report the C11 instrumentation.

Usage: python analyze_log.py <log.json|log_dir> [--dump-failures out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter


def load(path):
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "*.json")), key=os.path.getmtime)
        path = cands[-1]
    with open(path) as f:
        return path, json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--dump-failures", default=None)
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    path, log = load(args.path)
    samples = log.get("samples") or []
    scores = log.get("results", {}).get("scores", [])
    print("log:", path)
    for s in scores:
        print("  metrics:", {k: v["value"] for k, v in s["metrics"].items()})
    print("  n samples:", len(samples))

    stop_reasons = Counter()
    n_after_answer = 0
    n_no_answer = 0
    n_garbage = 0
    out_tokens = []
    fails = []
    for s in samples:
        out = s.get("output", {})
        choices = out.get("choices") or [{}]
        comp = (choices[0].get("message", {}).get("content") or "")
        if isinstance(comp, list):
            comp = "".join(c.get("text", "") for c in comp if isinstance(c, dict))
        stop_reasons[choices[0].get("stop_reason")] += 1
        usage = out.get("usage") or {}
        if usage.get("output_tokens"):
            out_tokens.append(usage["output_tokens"])
        m = re.search(r"ANSWER:\s*[^\n]*", comp)
        if not m:
            n_no_answer += 1
        elif comp[m.end():].strip():
            n_after_answer += 1
        if comp[:8].strip().startswith("!!!"):
            n_garbage += 1
        sc = list((s.get("scores") or {}).values())
        val = sc[0]["value"] if sc else None
        if val != "C":
            fails.append(
                {
                    "id": s.get("id"),
                    "target": s.get("target"),
                    "answer": sc[0].get("answer") if sc else None,
                    "completion": comp,
                }
            )
    n = max(len(samples), 1)
    print(f"  stop reasons: {dict(stop_reasons)}")
    print(f"  text after ANSWER line: {n_after_answer} ({n_after_answer/n:.1%})")
    print(f"  no ANSWER line at all:  {n_no_answer} ({n_no_answer/n:.1%})")
    print(f"  garbage '!!!' prefix:   {n_garbage}")
    if out_tokens:
        out_tokens.sort()
        print(
            f"  output tokens p50={out_tokens[len(out_tokens)//2]} "
            f"p95={out_tokens[int(len(out_tokens)*0.95)]} max={out_tokens[-1]}"
        )
    print(f"  incorrect: {len(fails)}")
    for f in fails[: args.show]:
        print("=" * 70)
        print("target", f["target"], "| extracted", f["answer"])
        print(f["completion"][:1500])
    if args.dump_failures:
        with open(args.dump_failures, "w") as fh:
            json.dump(fails, fh, indent=2)
        print("wrote", args.dump_failures)


if __name__ == "__main__":
    main()
