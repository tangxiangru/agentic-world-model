#!/usr/bin/env python3
"""Build SFT data for GSM8K from OpenMathInstruct-2 (gsm8k / augmented_gsm8k / math)."""
import argparse, glob, json, random, re, collections

import pyarrow.parquet as pq
import pyarrow.compute as pc

OMI = sorted(glob.glob(
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"))

BOXED = re.compile(r"\\boxed\{")


def strip_boxed(s: str) -> str:
    """Replace every \\boxed{...} with its content (balanced-brace aware)."""
    out = []
    i = 0
    while True:
        m = BOXED.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.start()])
        j = m.end()
        depth = 1
        while j < len(s) and depth:
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append(s[m.end():j])
        i = j + 1
    return "".join(out)


def clean_solution(sol: str, ans: str) -> str | None:
    sol = strip_boxed(sol).strip()
    if not sol:
        return None
    # drop any trailing "The answer is ..." style line, we append our own
    lines = [l.rstrip() for l in sol.split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    sol = "\n".join(lines).strip()
    return sol + f"\n\nANSWER: {ans}"


def load_omi(max_per_problem, want):
    """want: dict source -> n examples"""
    buckets = {k: [] for k in want}
    seen = collections.Counter()
    rows = []
    for f in OMI:
        t = pq.read_table(f)
        srcs = t.column("problem_source").to_pylist()
        probs = t.column("problem").to_pylist()
        sols = t.column("generated_solution").to_pylist()
        answ = t.column("expected_answer").to_pylist()
        for s, p, so, a in zip(srcs, probs, sols, answ):
            if s in buckets:
                rows.append((s, p, so, a))
        del t
    random.shuffle(rows)
    for s, p, so, a in rows:
        if len(buckets[s]) >= want[s]:
            continue
        key = (s, p)
        if seen[key] >= max_per_problem:
            continue
        c = clean_solution(so, a)
        if c is None:
            continue
        seen[key] += 1
        buckets[s].append({"question": p.strip(), "response": c,
                           "answer": a, "source": s})
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--n-gsm8k", type=int, default=14764)
    ap.add_argument("--n-aug-gsm8k", type=int, default=60000)
    ap.add_argument("--n-math", type=int, default=6000)
    ap.add_argument("--n-aug-math", type=int, default=8000)
    ap.add_argument("--max-per-problem", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    want = {"gsm8k": args.n_gsm8k, "augmented_gsm8k": args.n_aug_gsm8k,
            "math": args.n_math, "augmented_math": args.n_aug_math}
    want = {k: v for k, v in want.items() if v > 0}
    buckets = load_omi(args.max_per_problem, want)
    data = [r for b in buckets.values() for r in b]
    random.shuffle(data)
    for k, v in buckets.items():
        print(k, len(v))
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(data), "->", args.out)


if __name__ == "__main__":
    main()
