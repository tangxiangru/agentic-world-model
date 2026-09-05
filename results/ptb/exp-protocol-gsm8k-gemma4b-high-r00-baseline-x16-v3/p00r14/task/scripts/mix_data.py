"""Combine formatted corpora into one shuffled SFT file.

Each argument is `path:n` (n rows sampled without replacement, or `all`) or
`path:n:repeat`. Rows are already in the {question, target, answer, src} shape
build_sft_data.py / build_metamath.py / rft_sample.py emit, so nothing is
reformatted here - this only selects and shuffles.
"""
import argparse
import json
import random
from collections import Counter


def load(spec, rng):
    parts = spec.split(":")
    path, want = parts[0], parts[1]
    repeat = int(parts[2]) if len(parts) > 2 else 1
    rows = [json.loads(l) for l in open(path)]
    if want != "all":
        n = int(want)
        if n < len(rows):
            rows = rng.sample(rows, n)
    return rows * repeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="path:n[:repeat], n or 'all'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dedup", action="store_true",
                    help="drop rows whose (question, target) pair already appeared")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for spec in args.sources:
        got = load(spec, rng)
        print(f"{spec}: {len(got)} rows", flush=True)
        rows += got

    if args.dedup:
        seen, keep = set(), []
        for r in rows:
            k = (r["question"], r["target"])
            if k in seen:
                continue
            seen.add(k)
            keep.append(r)
        print(f"dedup: {len(rows)} -> {len(keep)}", flush=True)
        rows = keep

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(Counter(r["src"] for r in rows), flush=True)
    print(f"wrote {len(rows)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
