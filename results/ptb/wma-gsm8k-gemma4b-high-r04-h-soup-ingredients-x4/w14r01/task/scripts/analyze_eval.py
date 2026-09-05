#!/usr/bin/env python3
"""Summarise an inspect-ai eval log: accuracy plus the failure diagnostics the
cards care about (did the model stop after 'ANSWER: N', or run past it?)."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re


def last_numeric(s: str):
    words = s.strip().split()
    for w in reversed(words):
        c = w.replace(",", "").replace("$", "").replace("*", "")
        if c.replace(".", "").isnumeric():
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="path to the inspect .json log")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-failures", type=int, default=0)
    args = ap.parse_args()

    path = args.log
    if path is None:
        cands = sorted(
            glob.glob(os.path.join(args.log_dir, "*.json")), key=os.path.getmtime
        )
        cands = [c for c in cands if os.path.getsize(c) > 10000]
        path = cands[-1]
    print(f"log: {path}")
    with open(path) as f:
        log = json.load(f)

    samples = log.get("samples") or []
    n = len(samples)
    correct = 0
    ends_with_answer = 0
    ran_past = 0
    empty = 0
    garbage = 0
    stop_reasons = {}
    lens = []
    failures = []
    for s in samples:
        sc = list(s.get("scores", {}).values())
        ok = bool(sc) and sc[0].get("value") == "C"
        correct += ok
        out = s.get("output", {})
        comp = ""
        try:
            comp = out["choices"][0]["message"]["content"]
            if isinstance(comp, list):
                comp = "".join(c.get("text", "") for c in comp)
            sr = out["choices"][0].get("stop_reason")
        except Exception:
            sr = None
        stop_reasons[sr] = stop_reasons.get(sr, 0) + 1
        lens.append(len(comp))
        if not comp.strip():
            empty += 1
        if comp.lstrip().startswith("!!!!"):
            garbage += 1
        tail = comp.strip()[-120:]
        if re.search(r"ANSWER:\s*\$?-?[\d,]+(\.\d+)?\.?\s*\Z", comp.strip()):
            ends_with_answer += 1
        elif "ANSWER:" in comp:
            ran_past += 1
        if not ok and len(failures) < args.dump_failures:
            failures.append(
                {
                    "id": s.get("id"),
                    "target": s.get("target"),
                    "answer": sc[0].get("answer") if sc else None,
                    "tail": tail,
                    "chars": len(comp),
                }
            )

    res = {
        "log": path,
        "n": n,
        "accuracy": correct / n if n else None,
        "ends_with_answer_line": ends_with_answer / n if n else None,
        "has_answer_but_ran_past": ran_past / n if n else None,
        "empty_completions": empty,
        "garbage_bang_prefix": garbage,
        "stop_reasons": stop_reasons,
        "mean_chars": sum(lens) / len(lens) if lens else 0,
        "max_chars": max(lens) if lens else 0,
    }
    print(json.dumps(res, indent=2))
    for f in failures:
        print("---", f["id"], "target=", f["target"], "answer=", f["answer"])
        print(f["tail"])
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": res, "failures": failures}, f, indent=2)


if __name__ == "__main__":
    main()
