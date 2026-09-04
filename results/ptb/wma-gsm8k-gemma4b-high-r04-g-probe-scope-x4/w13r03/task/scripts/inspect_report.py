#!/usr/bin/env python3
"""Read an inspect_ai json eval log and report accuracy plus the format
diagnostics the score alone hides: how often the completion never terminated,
how often it lacked a trailing 'ANSWER:' line, and the output-token profile.

Usage: python scripts/inspect_report.py logs/<file>.json [--dump out.jsonl]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from glob import glob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=None)
    ap.add_argument("--dump", default=None, help="write per-sample records here")
    args = ap.parse_args()

    path = args.log
    if path is None:
        cands = sorted(glob("logs/*.json"))
        if not cands:
            sys.exit("no logs/*.json")
        path = cands[-1]
    d = json.load(open(path))

    samples = d.get("samples") or []
    n = len(samples)
    correct = 0
    no_answer_line = 0
    unterminated = 0
    out_tokens = []
    recs = []
    for s in samples:
        score = list(s.get("scores", {}).values())
        val = score[0]["value"] if score else None
        ok = val == "C"
        correct += ok
        msg = s["messages"][-1]
        text = msg.get("content")
        if isinstance(text, list):
            text = "".join(c.get("text", "") for c in text if isinstance(c, dict))
        text = text or ""
        out = s.get("output") or {}
        choices = out.get("choices") or []
        # inspect stores the stop reason on the choice, not on the output
        stop = (choices[0].get("stop_reason") if choices else None) or out.get("stop_reason")
        usage = out.get("usage") or {}
        ntok = usage.get("output_tokens")
        if ntok is None:
            ntok = usage.get("completion_tokens")
        out_tokens.append(ntok or 0)
        if stop != "stop":
            unterminated += 1
        tail = text.strip().splitlines()[-1] if text.strip() else ""
        if not re.match(r"^\s*ANSWER:", tail):
            no_answer_line += 1
        recs.append(
            {
                "id": s.get("id"),
                "correct": ok,
                "target": s.get("target"),
                "answer": score[0].get("answer") if score else None,
                "stop_reason": stop,
                "completion_tokens": usage.get("completion_tokens"),
                "tail": tail[:200],
                "completion": text,
            }
        )

    out_tokens.sort()
    p = lambda q: out_tokens[min(int(len(out_tokens) * q), len(out_tokens) - 1)] if out_tokens else 0
    print(f"log                : {path}")
    print(f"n                  : {n}")
    print(f"accuracy           : {correct / n:.4f}  ({correct}/{n})")
    print(f"no trailing ANSWER:: {no_answer_line} ({no_answer_line / n:.2%})")
    print(f"stop_reason != stop: {unterminated} ({unterminated / n:.2%})")
    print(f"completion tokens  : p50 {p(0.5)}  p90 {p(0.9)}  max {out_tokens[-1] if out_tokens else 0}")

    if args.dump:
        with open(args.dump, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"dumped per-sample records to {args.dump}")


if __name__ == "__main__":
    main()
