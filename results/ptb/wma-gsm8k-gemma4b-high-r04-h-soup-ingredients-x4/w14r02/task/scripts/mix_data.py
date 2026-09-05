#!/usr/bin/env python3
"""Mix jsonl SFT files, optionally excluding questions already used, and shuffle."""
from __future__ import annotations

import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--add", action="append", default=[], metavar="PATH[:N]",
                    help="file to include, optionally capped at N rows")
    ap.add_argument("--exclude-questions-from", action="append", default=[],
                    help="jsonl whose 'question' values are dropped from LATER --add files")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    excl = set()
    for p in a.exclude_questions_from:
        for line in open(p):
            q = json.loads(line).get("question")
            if q:
                excl.add(q.strip())
    print(f"{len(excl)} excluded questions")

    rng = random.Random(a.seed)
    rows = []
    for spec in a.add:
        path, _, cap = spec.partition(":")
        cap = int(cap) if cap else 0
        keep_excl = path.endswith("!")  # not used; kept simple
        got = []
        for line in open(path.rstrip("!")):
            r = json.loads(line)
            if r.get("question", "").strip() in excl:
                continue
            got.append(r)
        rng.shuffle(got)
        if cap:
            got = got[:cap]
        print(f"{path}: {len(got)} rows")
        rows.extend(got)

    rng.shuffle(rows)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
