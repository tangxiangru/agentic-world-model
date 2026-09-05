#!/usr/bin/env python3
"""Summarise an inspect_ai gsm8k eval log: accuracy plus format-failure diagnostics."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="inspect .json log; default = newest in logs/")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-failures", type=int, default=0)
    args = ap.parse_args()

    path = args.log or max(glob.glob("logs/*.json"), key=os.path.getmtime)
    d = json.load(open(path))
    samples = d.get("samples", [])
    n = len(samples)
    correct = 0
    no_answer_line = 0
    truncated = 0
    fails = []
    for s in samples:
        score = list(s["scores"].values())[0]
        ok = score["value"] == "C"
        correct += ok
        comp = ""
        for m in reversed(s.get("messages", [])):
            if m.get("role") == "assistant":
                c = m.get("content")
                comp = c if isinstance(c, str) else " ".join(
                    p.get("text", "") for p in c if isinstance(p, dict))
                break
        last = comp.rstrip().split("\n")[-1] if comp.strip() else ""
        if not re.match(r"^\s*ANSWER:\s*-?[\d,\.]+\s*$", last):
            no_answer_line += 1
        out = s.get("output", {}) or {}
        choices = out.get("choices") or []
        stop_reason = (choices[0].get("stop_reason") if choices else None) or out.get("stop_reason")
        if stop_reason in ("max_tokens", "length"):
            truncated += 1
        if not ok and len(fails) < args.dump_failures:
            fails.append({"id": s.get("id"), "target": s.get("target"),
                          "answer": score.get("answer"), "tail": comp[-600:]})

    res = {
        "log": path, "n": n, "accuracy": correct / max(n, 1),
        "no_answer_line_share": no_answer_line / max(n, 1),
        "truncated_share": truncated / max(n, 1),
    }
    print(json.dumps(res, indent=2))
    for f in fails:
        print("=" * 70)
        print(f["id"], "gold:", f["target"], "read:", f["answer"])
        print(f["tail"])
    if args.out:
        json.dump({**res, "failures": fails}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
