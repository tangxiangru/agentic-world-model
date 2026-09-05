#!/usr/bin/env python3
"""Run the official protocol on a checkpoint and tag its failures.

Usage: python scripts/eval_and_tag.py <model_path> <tag> [--limit 150]
Writes eval/<tag>_dev150.json (metrics) and analysis/<tag>_tags.json
(per-sample stop_reason / format tags, ids only - no benchmark text).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

TASK = "/home/ben/task"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("tag")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--max-connections", type=int, default=32)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    a = ap.parse_args()

    metrics = f"{TASK}/eval/{a.tag}_dev{a.limit}.json"
    before = set(glob.glob(f"{TASK}/logs/*.json"))
    cmd = [sys.executable, "evaluate.py", "--model-path", a.model_path,
           "--limit", str(a.limit), "--json-output-file", metrics,
           "--max-connections", str(a.max_connections),
           "--gpu-memory-utilization", str(a.gpu_memory_utilization)]
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    with open(f"{TASK}/logs/{a.tag}_eval.log", "w") as lf:
        r = subprocess.run(cmd, cwd=TASK, stdout=lf, stderr=subprocess.STDOUT)
    print(f"evaluate.py exit {r.returncode} in {(time.time()-t0)/60:.1f} min", flush=True)
    if not os.path.exists(metrics):
        print("NO METRICS FILE - see log")
        sys.exit(1)
    print(open(metrics).read())

    new = [p for p in glob.glob(f"{TASK}/logs/*.json") if p not in before]
    if not new:
        return
    logp = max(new, key=os.path.getmtime)
    d = json.load(open(logp))
    rows = []
    for s in d["samples"]:
        ch = s["output"]["choices"][0]
        c = ch["message"]["content"]
        if not isinstance(c, str):
            c = " ".join(x.get("text", "") for x in c)
        m = re.search(r"ANSWER:\s*\$?(-?[\d,\.]+)", c)
        rows.append(dict(id=s["id"], correct=s["scores"]["match"]["value"] == "C",
                         stop_reason=ch.get("stop_reason"),
                         has_answer_line=bool(m), n_chars=len(c),
                         n_answer_lines=len(re.findall(r"ANSWER:", c))))
    n = len(rows)
    summary = dict(
        eval_log=logp, n=n,
        accuracy=sum(r["correct"] for r in rows) / n,
        max_tokens_share=sum(r["stop_reason"] == "max_tokens" for r in rows) / n,
        no_answer_line_share=sum(not r["has_answer_line"] for r in rows) / n,
        multi_answer_line_share=sum(r["n_answer_lines"] > 1 for r in rows) / n,
        median_chars=sorted(r["n_chars"] for r in rows)[n // 2])
    out = f"{TASK}/analysis/{a.tag}_tags.json"
    json.dump(dict(summary=summary, rows=rows), open(out, "w"), indent=1)
    with open(f"{TASK}/analysis/{a.tag}_watch.jsonl", "w") as f:
        for r in rows:
            if not r["correct"]:
                f.write(json.dumps({"id": r["id"]}) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
