#!/usr/bin/env python3
"""Round-2 SFT mixture: fresh OMI2 samples (not used in round 1) + on-policy RFT data."""
import argparse, glob, json, random, re, collections
import pyarrow.parquet as pq
from prep_data import unbox, clean_ans, is_num, make, load_gsm8k_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", default="data/sft_v2.jsonl")
    ap.add_argument("--rft", default="data/rft_r1.jsonl")
    ap.add_argument("--out", default="data/sft_v3.jsonl")
    ap.add_argument("--n-gsm8k-omi", type=int, default=16000)
    ap.add_argument("--n-aug-gsm8k", type=int, default=46000)
    ap.add_argument("--n-math", type=int, default=4000)
    ap.add_argument("--n-aug-math", type=int, default=8000)
    ap.add_argument("--rft-repeat", type=int, default=1)
    ap.add_argument("--human-repeat", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(7)

    prev = set()
    for l in open(args.prev):
        r = json.loads(l)
        prev.add((r["question"].strip(), r["solution"]))

    caps = {"gsm8k": args.n_gsm8k_omi, "augmented_gsm8k": args.n_aug_gsm8k,
            "math": args.n_math, "augmented_math": args.n_aug_math}
    per_problem = collections.Counter()
    pcaps = {"gsm8k": 4, "augmented_gsm8k": 2, "math": 2, "augmented_math": 1}
    buckets = collections.defaultdict(list)
    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"))
    for f in files:
        t = pq.read_table(f)
        for p, s, a, src in zip(t.column("problem").to_pylist(),
                                t.column("generated_solution").to_pylist(),
                                t.column("expected_answer").to_pylist(),
                                t.column("problem_source").to_pylist()):
            if len(buckets[src]) >= caps.get(src, 0):
                continue
            a = clean_ans(a)
            if not is_num(a) or "\\boxed{" not in s:
                continue
            if len(s) > 2600 or len(s) < 40 or len(p) > 1600:
                continue
            body = unbox(s).strip()
            if "[asy]" in body or "\\begin{tabular}" in body:
                continue
            ex = make(p, body, a)
            if (ex["question"], ex["solution"]) in prev:
                continue
            key = (src, p)
            if per_problem[key] >= pcaps.get(src, 1):
                continue
            per_problem[key] += 1
            buckets[src].append(ex)
        if all(len(buckets[k]) >= v for k, v in caps.items()):
            break

    rows = []
    for k, v in buckets.items():
        print(k, len(v))
        rows += v

    gt = load_gsm8k_train()
    rows += gt * args.human_repeat
    print("gsm8k_human", len(gt) * args.human_repeat)

    nr = 0
    for l in open(args.rft):
        r = json.loads(l)
        sol = r["solution"].strip()
        rows += [{"question": r["question"], "solution": sol, "answer": r["answer"]}] * args.rft_repeat
        nr += args.rft_repeat
    print("rft", nr)

    seen, ded = set(), []
    for r in rows:
        ded.append(r)
    rng.shuffle(ded)
    print("total", len(ded))
    with open(args.out, "w") as f:
        for r in ded:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
