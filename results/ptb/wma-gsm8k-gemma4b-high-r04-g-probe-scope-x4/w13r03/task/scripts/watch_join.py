#!/usr/bin/env python3
"""Join a watch set of previously-failing item ids against a new eval dump.

  python scripts/watch_join.py analysis/exp-02_watch.jsonl analysis/exp-03_samples.jsonl \
      --baseline analysis/exp-02_e2_samples.jsonl
"""
from __future__ import annotations

import argparse
import json


def load(path: str) -> dict:
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("watch")
    ap.add_argument("new")
    ap.add_argument("--baseline", required=True)
    args = ap.parse_args()

    watch = [json.loads(l)["id"] for l in open(args.watch)]
    new = load(args.new)
    base = load(args.baseline)

    missing = [i for i in watch if i not in new]
    fixed = [i for i in watch if i in new and new[i]["correct"]]
    still = [i for i in watch if i in new and not new[i]["correct"]]
    regressions = [
        i for i, r in base.items() if r["correct"] and i in new and not new[i]["correct"]
    ]

    print(f"watch set n={len(watch)}  (unjoinable ids: {len(missing)})")
    print(f"fixed         : {len(fixed)}")
    print(f"still failing : {len(still)}")
    print(f"regressions   : {len(regressions)} (correct in baseline, wrong now)")
    print(f"net           : {len(fixed) - len(regressions):+d}")
    if regressions:
        print("regressed ids:", ", ".join(regressions[:20]))


if __name__ == "__main__":
    main()
