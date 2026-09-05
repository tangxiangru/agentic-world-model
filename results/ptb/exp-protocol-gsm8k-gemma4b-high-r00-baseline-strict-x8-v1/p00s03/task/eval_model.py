#!/usr/bin/env python3
"""Run the harness eval on a checkpoint and write accuracy + diagnostics.

Always the same protocol: evaluate.py --limit N --max-connections 32.
Writes eval/<tag>.json (harness metrics) and analysis/<tag>_diag.json
(non-stop share, first-ANSWER-line accuracy, mean completion tokens).
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

TASK = os.path.dirname(os.path.abspath(__file__))


def newest_log(after: float) -> str | None:
    logs = [p for p in glob.glob(os.path.join(TASK, "logs", "*_gsm8k_*.json"))
            if os.path.getmtime(p) > after]
    return max(logs, key=os.path.getmtime) if logs else None


def diagnostics(log_path: str) -> dict:
    d = json.load(open(log_path))
    s = d["samples"]
    ok = nostop = 0
    toks = []
    for x in s:
        ch = x["output"]["choices"][0]
        txt = ch["message"]["content"]
        if ch.get("stop_reason") != "stop":
            nostop += 1
        m = re.findall(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", txt)
        a = m[0].replace(",", "") if m else None
        t = x["target"].replace(",", "")
        try:
            ok += a is not None and abs(float(a) - float(t)) < 1e-6
        except ValueError:
            pass
        u = x["output"].get("usage") or {}
        toks.append(u.get("completion_tokens") or 0)
    n = len(s)
    return {
        "n": n,
        "graded_accuracy": sum(1 for x in s if x["scores"]["match"]["value"] == "C") / n,
        "first_answer_line_accuracy": ok / n,
        "non_stop_share": nostop / n,
        "mean_completion_tokens": sum(toks) / n,
        "log": log_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--max-connections", type=int, default=32)
    args = ap.parse_args()

    out = os.path.join(TASK, "eval", f"{args.tag}.json")
    t0 = time.time()
    cmd = [sys.executable, "evaluate.py", "--model-path", args.model_path,
           "--limit", str(args.limit), "--max-connections", str(args.max_connections),
           "--json-output-file", out]
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=TASK)
    if r.returncode != 0:
        print("evaluate.py failed", r.returncode)
        sys.exit(r.returncode)
    log = newest_log(t0)
    diag = diagnostics(log) if log else {}
    diag["model_path"] = args.model_path
    diag["limit"] = args.limit
    diag["wall_s"] = round(time.time() - t0, 1)
    dp = os.path.join(TASK, "analysis", f"{args.tag}_diag.json")
    json.dump(diag, open(dp, "w"), indent=1)
    print(json.dumps(diag, indent=1))


if __name__ == "__main__":
    main()
