#!/usr/bin/env python3
"""Run evaluate.py under the locked protocol and summarise the failure modes.

Writes eval/<tag>.json (metrics), eval/<tag>_log.json (the inspect log) and
analysis/<tag>_diag.json (stop reasons, format failures, per-item results).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

TASK = "/home/ben/task"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--max-connections", type=int, default=16)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    metrics_path = os.path.join(TASK, "eval", f"{args.tag}.json")
    t0 = time.time()
    cmd = [
        sys.executable, "evaluate.py",
        "--model-path", args.model_path,
        "--limit", str(args.limit),
        "--max-connections", str(args.max_connections),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--json-output-file", metrics_path,
    ]
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=TASK)
    print(f"[eval] exit {r.returncode} in {(time.time() - t0) / 60:.1f} min", flush=True)
    if not os.path.exists(metrics_path):
        sys.exit(f"no metrics written for {args.tag}")

    logf = max(glob.glob(os.path.join(TASK, "logs", "*_gsm8k_*.json")), key=os.path.getmtime)
    dest = os.path.join(TASK, "eval", f"{args.tag}_log.json")
    shutil.copy(logf, dest)

    d = json.load(open(dest))
    samples = d["samples"]
    stops, items = {}, []
    no_marker = 0
    for s in samples:
        ch = s["output"]["choices"][0]
        txt = ch["message"]["content"]
        sr = ch.get("stop_reason")
        stops[sr] = stops.get(sr, 0) + 1
        if "ANSWER:" not in txt:
            no_marker += 1
        items.append({
            "id": s["id"],
            "correct": s["scores"]["match"]["value"] == "C",
            "gold": s["target"],
            "answer": s["scores"]["match"].get("answer"),
            "stop_reason": sr,
            "n_chars": len(txt),
        })
    diag = {
        "tag": args.tag,
        "metrics": json.load(open(metrics_path)),
        "eval_log": dest,
        "n": len(samples),
        "stop_reasons": stops,
        "stopped_share": stops.get("stop", 0) / max(1, len(samples)),
        "no_answer_marker_share": no_marker / max(1, len(samples)),
        "median_chars": sorted(x["n_chars"] for x in items)[len(items) // 2],
        "items": items,
    }
    out = os.path.join(TASK, "analysis", f"{args.tag}_diag.json")
    json.dump(diag, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in diag.items() if k != "items"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
