#!/usr/bin/env python3
"""Score several candidate checkpoints under one identical protocol and print
the ranking. Every candidate is run through the same evaluate.py invocation,
sequentially, on the same slice - the point is that the arms are comparable,
not that it is fast.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

TASK = "/home/ben/task"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max-connections", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    assert len(args.candidates) == len(args.tags)
    results = {}
    for path, tag in zip(args.candidates, args.tags):
        out = f"{TASK}/eval/{tag}_dev{args.limit}.json"
        log = f"{TASK}/logs/{tag}_dev{args.limit}.log"
        cmd = [
            sys.executable, "evaluate.py",
            "--model-path", path,
            "--limit", str(args.limit),
            "--max-connections", str(args.max_connections),
            "--max-tokens", str(args.max_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--json-output-file", out,
        ]
        print(f"[select] {tag}: {' '.join(cmd)}", flush=True)
        with open(log, "w") as f:
            rc = subprocess.call(cmd, cwd=TASK, stdout=f, stderr=subprocess.STDOUT)
        if rc != 0 or not os.path.exists(out):
            print(f"[select] {tag}: FAILED rc={rc}, see {log}", flush=True)
            results[tag] = None
            continue
        results[tag] = json.load(open(out))
        print(f"[select] {tag}: {results[tag]}", flush=True)

    print("\n[select] ranking at n=%d" % args.limit)
    for tag, r in sorted(
        results.items(), key=lambda kv: -(kv[1]["accuracy"] if kv[1] else -1)
    ):
        if r:
            print(f"  {tag:24s} accuracy {r['accuracy']:.4f}  stderr {r['stderr']:.4f}")
        else:
            print(f"  {tag:24s} FAILED")


if __name__ == "__main__":
    main()
