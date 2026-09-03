#!/usr/bin/env python3
"""Build a pool file from microsoft/orca-math-word-problems-200k.

Orca-Math has no separate answer field: the gold number is whatever the solution
text ends on. That happens to be exactly what this benchmark's grader reads (the
last number of the completion), but it means the label is only trustworthy when
the solution really does conclude with its answer. Rows are kept only when a
number appears near the very end of the text.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

sys.path.insert(0, "/home/ben/task/scripts")
from format_utils import MATH_PROMPT_TEMPLATE  # noqa: E402

NUMLIKE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def clean(text: str) -> str:
    t = BOXED.sub(r"\1", text)
    t = t.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    t = re.sub(r"\*\*", "", t)
    return t.strip()


def last_number_with_pos(text: str):
    """Return (normalised number, char offset of the token) for the last numeric token."""
    best = None
    for m in re.finditer(r"-?\d[\d,]*(?:\.\d+)?", text):
        best = m
    if best is None:
        return None, None
    c = best.group(0).replace(",", "")
    if c.endswith("."):
        c = c[:-1]
    if not NUMLIKE.match(c):
        return None, None
    if "." in c:
        c = c.rstrip("0").rstrip(".") or "0"
    return c, best.start()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/pool_orca.jsonl")
    ap.add_argument("--tail-window", type=int, default=120,
                    help="the answer number must start within this many chars of the end")
    ap.add_argument("--max-chars", type=int, default=2600)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    seen: set[str] = set()
    n = kept = 0
    drop_tail = drop_num = drop_long = drop_dup = 0
    with open(args.out, "w") as out:
        for r in ds:
            n += 1
            q = r["question"].strip()
            a = clean(r["answer"])
            if q in seen:
                drop_dup += 1
                continue
            if len(a) > args.max_chars or len(q) > 1500 or len(a) < 20:
                drop_long += 1
                continue
            num, pos = last_number_with_pos(a)
            if num is None:
                drop_num += 1
                continue
            if len(a) - pos > args.tail_window:
                drop_tail += 1
                continue
            seen.add(q)
            completion = f"{a}\nANSWER: {num}"
            if completion.count("ANSWER:") != 1:
                continue
            out.write(json.dumps({
                "problem": q,
                "prompt": MATH_PROMPT_TEMPLATE.format(prompt=q),
                "completion": completion,
                "answer": num,
                "source": "orca_math",
            }) + "\n")
            kept += 1
    print(f"read {n}, kept {kept}; dropped dup {drop_dup}, long {drop_long}, "
          f"no-number {drop_num}, number-not-at-end {drop_tail} -> {args.out}")


if __name__ == "__main__":
    main()
