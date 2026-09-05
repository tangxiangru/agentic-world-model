#!/usr/bin/env python3
"""Diagnose an inspect-ai gsm8k eval log: graded accuracy vs format-blind accuracy.

The grader reads the LAST numeric word of the completion. This tells you how much
of the gap is reasoning and how much is output shape, by also scoring the FIRST
'ANSWER:' line the model emits, and by counting stop reasons.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter


def norm(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".").rstrip("%")
    m = re.match(r"^-?\d+(\.\d+)?$", s)
    if not m:
        return None
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        return s


def first_answer_line(text: str) -> str | None:
    for m in re.finditer(r"ANSWER:\s*([^\n]*)", text):
        tok = m.group(1).strip().split()
        if tok:
            v = norm(tok[0])
            if v is not None:
                return v
    return None


def last_number(text: str) -> str | None:
    words = re.split(r"\s+", text.strip())
    for w in reversed(words):
        v = norm(w)
        if v is not None:
            return v
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--dump-wrong", default=None)
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    d = json.load(open(args.log))
    samples = d["samples"]
    graded = fmt = 0
    stops = Counter()
    lens = []
    wrong = []
    for s in samples:
        comp = s["output"]["choices"][0]["message"]["content"]
        if isinstance(comp, list):
            comp = "".join(c.get("text", "") for c in comp)
        stops[s["output"]["choices"][0].get("stop_reason")] += 1
        u = s["output"].get("usage") or {}
        lens.append(u.get("completion_tokens") or u.get("output_tokens") or 0)
        gold = norm(s["target"]) or s["target"]
        ok = s["scores"]["match"]["value"] == "C"
        graded += ok
        fa = first_answer_line(comp)
        fmt += fa == gold
        if not ok:
            wrong.append(
                {
                    "id": s["id"],
                    "question": s["input"] if isinstance(s["input"], str) else str(s["input"])[:400],
                    "gold": gold,
                    "first_answer": fa,
                    "last_number": last_number(comp),
                    "stop_reason": s["output"]["choices"][0].get("stop_reason"),
                    "completion_tail": comp[-600:],
                    "completion_head": comp[:600],
                }
            )
    n = len(samples)
    print(f"n={n}")
    print(f"graded accuracy            : {graded/n:.4f}  ({graded})")
    print(f"first-ANSWER-line accuracy : {fmt/n:.4f}  ({fmt})")
    print(f"gap attributable to shape  : {(fmt-graded)/n:.4f}")
    print("stop reasons:", dict(stops))
    print(f"completion tokens: mean={sum(lens)/n:.0f} max={max(lens)}")
    no_ans = sum(1 for w in wrong if w["first_answer"] is None)
    print(f"wrong items with no parseable 'ANSWER:' line at all: {no_ans}/{len(wrong)}")
    if args.dump_wrong:
        with open(args.dump_wrong, "w") as f:
            for w in wrong:
                f.write(json.dumps(w) + "\n")
        print("wrote", args.dump_wrong)
    for w in wrong[: args.show]:
        print("=" * 70)
        print(json.dumps(w, indent=2)[:2500])


if __name__ == "__main__":
    main()
