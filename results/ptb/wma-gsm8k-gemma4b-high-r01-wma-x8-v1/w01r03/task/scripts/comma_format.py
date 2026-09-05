#!/usr/bin/env python3
"""Rewrite the final 'ANSWER: n' line of an SFT jsonl to use thousands separators.

Why: inspect_ai's match(numeric=True, location='end') only takes its numeric
path when target.isnumeric() is true.  A GSM8K gold answer written '5,600' is
not isnumeric(), so the scorer falls back to a *string* comparison and demands
the completion end with the literal '5,600'.  A completion ending '5600' is
marked wrong even though the value is right (4 of 500 items at n=500).

Emitting the separator is safe in the other direction: when the gold is '5600'
the numeric path runs strip_numeric_punctuation over the completion first, so
'5,600' normalises to '5600' and still matches.  Verified against
inspect_ai.scorer._common.match_str for both cases.  No GSM8K gold answer
contains a decimal point (0 of 7473 in the train split), which is the one case
where a separator would hurt.
"""
from __future__ import annotations

import argparse
import json
import re

STOP_TOKEN = "<end_of_turn>"
ANSWER_LINE = re.compile(r"ANSWER:\s*(-?[\d,]+(?:\.\d+)?)\s*(<end_of_turn>)?\s*$")


def group(n: str) -> str:
    neg = n.startswith("-")
    n = n.lstrip("-").replace(",", "")
    return ("-" if neg else "") + f"{int(n):,}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N rows of each input")
    ap.add_argument("--take", type=int, default=None, help="take at most N rows of each input")
    args = ap.parse_args()

    n_out = n_changed = n_skipped = 0
    with open(args.out, "w") as g:
        for path in args.inp:
            rows = [json.loads(line) for line in open(path)]
            rows = rows[args.skip:]
            if args.take:
                rows = rows[: args.take]
            for r in rows:
                c = r["completion"]
                m = ANSWER_LINE.search(c)
                if not m or "." in m.group(1):
                    n_skipped += 1
                    g.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_out += 1
                    continue
                new = group(m.group(1))
                if new != m.group(1):
                    n_changed += 1
                r["completion"] = c[: m.start()] + f"ANSWER: {new}" + STOP_TOKEN
                g.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_out += 1
    print(json.dumps({"rows": n_out, "answer_lines_regrouped": n_changed,
                      "left_alone": n_skipped, "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
