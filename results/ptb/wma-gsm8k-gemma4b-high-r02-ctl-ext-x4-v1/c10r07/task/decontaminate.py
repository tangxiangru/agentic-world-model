#!/usr/bin/env python3
"""Stronger decontamination than ../contamination_check.py can give on GSM8K.

The provided checker early-exits on documents that yield fewer than 20 sampled
n-grams; GSM8K questions are ~60 tokens, so it scores 0 matches even when the
input IS a verbatim test item (verified: feeding it 5 rows of ../test_data.json
returns "Total matches 0"). This script therefore does the check directly:
word-level 8-gram containment of every training problem against all 1319 test
questions, via an inverted index.

  containment(train, test) = |ngrams(train) & ngrams(test)| / |ngrams(test)|

Anything at or above --threshold is written to the report and dropped.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

WORD = re.compile(r"[a-z0-9]+")


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = WORD.findall(text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="/home/ben/test_data.json")
    ap.add_argument("--input", required=True)          # jsonl with a "question" field
    ap.add_argument("--paired", default=None)          # same-order jsonl to filter alongside
    ap.add_argument("--out", default=None)
    ap.add_argument("--paired-out", default=None)
    ap.add_argument("--report", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.30)
    args = ap.parse_args()

    ref = json.load(open(args.reference))
    ref_ngrams = [ngrams(r["question"], args.n) for r in ref]
    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, g in enumerate(ref_ngrams):
        for ng in g:
            index[ng].append(i)
    print(f"indexed {len(ref)} test questions, {len(index)} unique {args.n}-grams")

    rows = [json.loads(l) for l in open(args.input)]
    paired = [json.loads(l) for l in open(args.paired)] if args.paired else None
    if paired is not None:
        assert len(paired) == len(rows), (len(paired), len(rows))

    flagged, worst = [], 0.0
    keep = []
    for j, r in enumerate(rows):
        g = ngrams(r["question"], args.n)
        hits: dict[int, int] = defaultdict(int)
        for ng in g:
            for i in index.get(ng, ()):
                hits[i] += 1
        score, which = 0.0, -1
        for i, c in hits.items():
            s = c / max(1, len(ref_ngrams[i]))
            if s > score:
                score, which = s, i
        worst = max(worst, score)
        if score >= args.threshold:
            flagged.append({"row": j, "score": round(score, 3),
                            "test_idx": which,
                            "train_question": r["question"][:300],
                            "test_question": ref[which]["question"][:300]})
        else:
            keep.append(j)

    with open(args.report, "w") as f:
        for x in flagged:
            f.write(json.dumps(x) + "\n")
    print(f"rows={len(rows)} flagged={len(flagged)} max_containment={worst:.3f} "
          f"threshold={args.threshold}")

    if args.out:
        with open(args.out, "w") as f:
            for j in keep:
                f.write(json.dumps(rows[j]) + "\n")
        print(f"wrote {len(keep)} clean rows to {args.out}")
    if args.paired_out:
        with open(args.paired_out, "w") as f:
            for j in keep:
                f.write(json.dumps(paired[j]) + "\n")
        print(f"wrote {len(keep)} clean rows to {args.paired_out}")

    raise SystemExit(1 if flagged else 0)


if __name__ == "__main__":
    main()
