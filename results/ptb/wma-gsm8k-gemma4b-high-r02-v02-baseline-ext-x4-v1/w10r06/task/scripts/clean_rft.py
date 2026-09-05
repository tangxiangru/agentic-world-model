#!/usr/bin/env python3
"""Truncate rejection-sampled completions at their first ANSWER line.

rft_sample.py asked vLLM to stop on the string "<end_of_turn>", but vLLM
detokenises with skip_special_tokens=True, so that string never appears in the
text and the offline generate() ran on to max_tokens. Many completions
therefore repeat "ANSWER: n" dozens of times - the double_answer_format
pitfall, and the answer_marker_single preflight check caught it. The reasoning
before the first ANSWER line is intact, so truncating there salvages the row.
"""
from __future__ import annotations

import argparse
import json
import re

STOP = "<end_of_turn>"
ANSWER_LINE = re.compile(r"ANSWER:\s*(-?\d[\d,]*(?:\.\d+)?)")


def norm(v: str) -> str:
    v = v.replace(",", "")
    if v.endswith(".0"):
        v = v[:-2]
    if "." in v:
        v = v.rstrip("0").rstrip(".")
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chars", type=int, default=3000)
    args = ap.parse_args()

    kept, dropped_no_answer, dropped_mismatch, dropped_long, truncated = 0, 0, 0, 0, 0
    rows = []
    for line in open(args.inp):
        r = json.loads(line)
        txt = r["target"]
        if txt.endswith(STOP):
            txt = txt[: -len(STOP)]
        m = ANSWER_LINE.search(txt)
        if not m:
            dropped_no_answer += 1
            continue
        if m.end() != len(txt.rstrip()):
            truncated += 1
        body = txt[: m.end()].rstrip()
        if norm(m.group(1)) != norm(r["answer"]):
            dropped_mismatch += 1
            continue
        if len(body) > args.max_chars or body.count("ANSWER: ") != 1:
            dropped_long += 1
            continue
        rows.append({"question": r["question"], "target": body + STOP,
                     "answer": r["answer"], "source": r["source"]})
        kept += 1

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    doc = args.out.replace(".jsonl", "_docs.jsonl")
    with open(doc, "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")
    print(json.dumps({"kept": kept, "truncated": truncated,
                      "dropped_no_answer": dropped_no_answer,
                      "dropped_first_answer_wrong": dropped_mismatch,
                      "dropped_long_or_multi": dropped_long}, indent=2))
    print("wrote", args.out, doc)


if __name__ == "__main__":
    main()
