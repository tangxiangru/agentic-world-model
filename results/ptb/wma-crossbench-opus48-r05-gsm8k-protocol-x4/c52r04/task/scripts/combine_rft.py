#!/usr/bin/env python3
"""Combine gold GSM8K-train SFT data with correct self-sampled (RFT) solutions."""
import argparse, json, re


def load(path):
    return [json.loads(l) for l in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/gsm8k_train_sft.jsonl")
    ap.add_argument("--rft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap_per_q", type=int, default=3, help="max RFT samples per prompt")
    args = ap.parse_args()

    gold = load(args.gold)
    rft = load(args.rft)

    rows = list(gold)  # always keep all gold solutions
    # cap RFT samples per prompt and dedup identical completions
    per_q, seen = {}, set()
    kept_rft = 0
    for r in rft:
        p = r["prompt"]
        key = (p, re.sub(r"\s+", " ", r["completion"]))
        if key in seen:
            continue
        if per_q.get(p, 0) >= args.cap_per_q:
            continue
        seen.add(key)
        per_q[p] = per_q.get(p, 0) + 1
        rows.append({"prompt": r["prompt"], "completion": r["completion"], "target": r["target"]})
        kept_rft += 1

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[combine] gold={len(gold)} rft_in={len(rft)} rft_kept={kept_rft} total={len(rows)}")


if __name__ == "__main__":
    main()
