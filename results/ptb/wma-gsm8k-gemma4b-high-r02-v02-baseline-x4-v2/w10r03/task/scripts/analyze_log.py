#!/usr/bin/env python3
"""Summarise an inspect-ai eval log: accuracy, stop reasons, completion lengths,
and the items that failed (for building a watch set)."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter


def latest_log(logdir="/home/ben/task/logs") -> str:
    cands = sorted(glob.glob(os.path.join(logdir, "**", "*.json"), recursive=True), key=os.path.getmtime)
    if not cands:
        raise SystemExit(f"no logs under {logdir}")
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=None)
    ap.add_argument("--dump-failures", default=None)
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()
    path = args.log or latest_log()
    with open(path) as f:
        log = json.load(f)

    samples = log.get("samples") or []
    stop = Counter()
    correct = 0
    lens = []
    failures = []
    for s in samples:
        sc = list((s.get("scores") or {}).values())
        ok = bool(sc) and sc[0].get("value") == "C"
        correct += ok
        out = s.get("output") or {}
        ch = (out.get("choices") or [{}])[0]
        stop[ch.get("stop_reason")] += 1
        txt = (ch.get("message") or {}).get("content") or ""
        if isinstance(txt, list):
            txt = "".join(p.get("text", "") for p in txt)
        lens.append((out.get("usage") or {}).get("completion_tokens") or 0)
        if not ok:
            failures.append({
                "id": s.get("id"),
                "question": s.get("input") if isinstance(s.get("input"), str) else str(s.get("input"))[:400],
                "gold": s.get("target"),
                "model_output": txt,
                "stop_reason": ch.get("stop_reason"),
            })

    n = len(samples)
    print(f"log: {path}")
    print(f"n={n}  correct={correct}  accuracy={correct/max(1,n):.4f}")
    print("stop_reason:", dict(stop))
    if lens:
        lens_s = sorted(lens)
        print(f"completion_tokens p50={lens_s[n//2]} p95={lens_s[int(n*0.95)-1]} max={lens_s[-1]}")
    if args.dump_failures:
        with open(args.dump_failures, "w") as f:
            for r in failures:
                f.write(json.dumps(r) + "\n")
        print("wrote", args.dump_failures, len(failures))
    for r in failures[: args.show]:
        print("=" * 70)
        print("GOLD:", r["gold"], "| stop:", r["stop_reason"])
        print(r["model_output"][:2500])


if __name__ == "__main__":
    main()
