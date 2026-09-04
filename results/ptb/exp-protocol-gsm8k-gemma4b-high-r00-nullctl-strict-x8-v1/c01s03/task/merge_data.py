#!/usr/bin/env python3
"""Merge the OpenMathInstruct SFT pool with self-generated (rejection-sampled) data."""
import argparse, json, random, collections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="work/sft1.jsonl")
    ap.add_argument("--rft", default="work/rft.jsonl")
    ap.add_argument("--out", default="work/sft2.jsonl")
    ap.add_argument("--n-sft-gsm", type=int, default=45000)
    ap.add_argument("--n-sft-math", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=1)
    # keep at most this many RFT solutions for problems the model already nails
    ap.add_argument("--easy-cap", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    sft = [json.loads(l) for l in open(args.sft)]
    # crude split: MATH-style items carry latex markers / non-word-problem phrasing
    gsmish, mathish = [], []
    for r in sft:
        (mathish if ("$" in r["question"] or "\\" in r["question"]) else gsmish).append(r)
    rng.shuffle(gsmish); rng.shuffle(mathish)
    out = gsmish[: args.n_sft_gsm] + mathish[: args.n_sft_math]
    print("sft pool: gsmish", len(gsmish), "-> ", min(len(gsmish), args.n_sft_gsm),
          "| mathish", len(mathish), "->", min(len(mathish), args.n_sft_math))

    rft = [json.loads(l) for l in open(args.rft)]
    per_q = collections.defaultdict(list)
    for r in rft:
        per_q[r["question"]].append(r)
    kept = 0
    for q, rs in per_q.items():
        nc = rs[0].get("n_correct", 1)
        k = rs[0].get("k", 4)
        cap = args.easy_cap if nc >= k else len(rs)
        for r in rs[:cap]:
            out.append({"question": r["question"], "solution": r["solution"],
                        "answer": r["answer"]})
            kept += 1
    print("rft problems", len(per_q), "kept solutions", kept)

    rng.shuffle(out)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(out), "->", args.out)


if __name__ == "__main__":
    main()
