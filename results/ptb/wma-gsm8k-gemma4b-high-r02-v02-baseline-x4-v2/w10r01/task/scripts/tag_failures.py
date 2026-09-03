#!/usr/bin/env python3
"""Coarse failure taxonomy over a scripts/gen.py output file.

Buckets are chosen to separate the things that need different fixes:
  no_stop        - generation hit the token cap: a turn-discipline defect
  no_marker      - stopped but never wrote 'ANSWER: ': a format defect
  gold_in_chain  - the right number is computed but not carried to the answer
  wrong_answer   - the chain reaches a different number: a reasoning defect
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from common import ANSWER_MARKER, extract_answer, graded_correct, norm_answer

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    tags = Counter()
    rows = []
    n = ok = 0
    for line in open(args.samples):
        r = json.loads(line)
        comp, fin = r["completions"][0], r["finish"][0]
        gold = norm_answer(str(r["gold"]))
        n += 1
        if graded_correct(comp, gold):
            ok += 1
            tags["correct"] += 1
            continue
        if fin != "stop":
            t = "no_stop"
        elif ANSWER_MARKER not in comp:
            t = "no_marker"
        else:
            body = comp.split(ANSWER_MARKER)[0]
            nums = {norm_answer(x) for x in NUM.findall(body)}
            t = "gold_in_chain" if gold in nums else "wrong_answer"
        tags[t] += 1
        rows.append({"id": r["id"], "tag": t, "gold": gold, "got": extract_answer(comp),
                     "question": r["question"], "completion": comp})

    print(json.dumps({"n": n, "accuracy": round(ok / n, 4), "tags": dict(tags),
                      "failure_share": {k: round(v / max(1, n - ok), 3)
                                        for k, v in tags.items() if k != "correct"}}, indent=2))
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
    for r in rows[: args.show]:
        print("=" * 70, r["tag"], "gold", r["gold"], "got", r["got"])
        print(r["question"][:300])
        print("---")
        print(r["completion"][:900])


if __name__ == "__main__":
    main()
