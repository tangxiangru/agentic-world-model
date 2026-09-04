#!/usr/bin/env python3
"""Parse an inspect-ai json eval log: accuracy, format/termination diagnostics.

evaluate.py --json-output-file writes only the metrics dict, so every
per-sample diagnostic has to come from the inspect log itself.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter

ANSWER_LINE = re.compile(r"ANSWER:\s*\$?-?[\d,]+(?:\.\d+)?\$?\s*$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="path to the inspect .json log")
    ap.add_argument("--logdir", default="/home/ben/task/logs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-wrong", type=int, default=0)
    args = ap.parse_args()

    path = args.log
    if path is None:
        cands = glob.glob(os.path.join(args.logdir, "**", "*.json"), recursive=True)
        cands = [c for c in cands if os.path.getsize(c) > 50_000]
        path = max(cands, key=os.path.getmtime)
    print("log:", path)
    log = json.load(open(path))

    samples = log.get("samples") or []
    n = len(samples)
    correct = 0
    no_answer_line = 0
    degenerate = 0
    at_cap = 0
    stop_reasons = Counter()
    lens = []
    wrong = []
    for s in samples:
        sc = list(s.get("scores", {}).values())
        ok = sc and sc[0].get("value") == "C"
        correct += bool(ok)
        out = s.get("output", {})
        choices = out.get("choices") or [{}]
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        sr = choices[0].get("stop_reason")
        stop_reasons[sr] += 1
        if sr == "max_tokens":
            at_cap += 1
        usage = out.get("usage") or {}
        lens.append(usage.get("completion_tokens") or 0)
        txt = content.strip()
        if not ANSWER_LINE.search(txt):
            no_answer_line += 1
        head = txt[:20]
        if head and (len(set(head)) <= 2 or head.startswith("!!")):
            degenerate += 1
        if not ok and len(wrong) < args.dump_wrong:
            wrong.append({
                "id": s.get("id"),
                "target": s.get("target"),
                "tail": txt[-400:],
                "stop_reason": sr,
            })

    res = {
        "log": path,
        "n": n,
        "accuracy": correct / n if n else None,
        "no_answer_line": no_answer_line,
        "no_answer_line_frac": no_answer_line / n if n else None,
        "degenerate": degenerate,
        "at_max_tokens": at_cap,
        "stop_reasons": dict(stop_reasons),
        "mean_completion_tokens": sum(lens) / n if n else None,
        "max_completion_tokens": max(lens) if lens else None,
        "model_args": (log.get("eval") or {}).get("model_args"),
        "generate_config": (log.get("eval") or {}).get("config"),
    }
    print(json.dumps(res, indent=2)[:4000])
    if wrong:
        print(json.dumps(wrong, indent=2)[:8000])
    if args.out:
        res["wrong_examples"] = wrong
        json.dump(res, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
