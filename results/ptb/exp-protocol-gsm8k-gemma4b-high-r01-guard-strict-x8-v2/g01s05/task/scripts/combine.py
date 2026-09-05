#!/usr/bin/env python3
"""Concatenate rendered {prompt, completion} jsonl files, optionally filtering by
`src` and repeating a file, then shuffle. Also writes a .contam.jsonl for the checker."""
from __future__ import annotations

import argparse
import hashlib
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--part",
        action="append",
        default=[],
        help="path[:src_filter][:xN] e.g. data/sft_mix.jsonl:gsm8k_gold:x1",
    )
    ap.add_argument("--exclude-prompts-from", default=None,
                    help="drop rows whose prompt appears in this jsonl (avoid re-training on seen rows)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    skip = set()
    if args.exclude_prompts_from:
        with open(args.exclude_prompts_from) as f:
            for line in f:
                skip.add(hashlib.sha1(json.loads(line)["prompt"].encode()).hexdigest())
        print("exclude set:", len(skip))

    rows = []
    for spec in args.part:
        bits = spec.split(":")
        path = bits[0]
        src_filter = bits[1] if len(bits) > 1 and bits[1] else None
        rep = int(bits[2][1:]) if len(bits) > 2 and bits[2].startswith("x") else 1
        got = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if src_filter and r.get("src") != src_filter:
                    continue
                if skip and hashlib.sha1(r["prompt"].encode()).hexdigest() in skip:
                    continue
                got.append({"prompt": r["prompt"], "completion": r["completion"],
                            "src": r.get("src", "?")})
        print(f"{path} src={src_filter} x{rep}: {len(got)} rows")
        for _ in range(rep):
            rows += got

    random.Random(args.seed).shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(args.out.replace(".jsonl", ".contam.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["prompt"] + r["completion"]}) + "\n")
    from collections import Counter
    print("total", len(rows), Counter(r["src"] for r in rows))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
