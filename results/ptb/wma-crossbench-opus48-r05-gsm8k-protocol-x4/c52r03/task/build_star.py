#!/usr/bin/env python3
"""Combine the exp-02 SFT corpus (metamath_gsm) with rejection-sampled RFT traces
into a single STaR corpus. All rows are {prompt, completion, gold, text} with a
single 'ANSWER: N' marker. Shuffle with a fixed seed. No test data touched.
"""
import argparse, json, random

def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metamath", default="data/metamath_gsm.jsonl")
    ap.add_argument("--rft", default="data/rft_exp02.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mm = load(args.metamath)
    rft = load(args.rft)
    # sanity: exactly one ANSWER marker, no '####'
    bad = 0
    for r in mm + rft:
        c = r["completion"]
        if c.count("ANSWER:") != 1 or "####" in c:
            bad += 1
    rows = mm + rft
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[build_star] metamath={len(mm)} rft={len(rft)} total={len(rows)} bad_marker={bad} -> {args.out}")

if __name__ == "__main__":
    main()
