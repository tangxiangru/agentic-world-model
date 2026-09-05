#!/usr/bin/env python3
"""Mix the rejection-sampled rows with a replay slice of the exp-02 SFT set.

Two reasons the mix is not RFT-only:
  * data/rft_v1.jsonl carries no few-shot system prefixes, and the harness
    always prompts with a 10-shot system block. exp-02 bought robustness to
    that with 8% prefixed rows; training on prefix-free data alone would decay
    it. 8% of the RFT rows get the same 1-3 shot treatment here.
  * a replay slice of sft_v2 keeps the 405B-written solutions in the mix, so
    the model is not narrowed onto only what it could already do.
"""
from __future__ import annotations

import argparse
import json
import random
import re


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", default="data/rft_v1.jsonl")
    ap.add_argument("--sft", default="data/sft_v2.jsonl")
    ap.add_argument("--replay", type=int, default=25000)
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    ap.add_argument("--out", default="data/rft_mix.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rft = [json.loads(l) for l in open(args.rft)]
    sft = [json.loads(l) for l in open(args.sft)]
    rng = random.Random(args.seed)

    dev = {norm_q(json.loads(l)["question"]) for l in open("data/dev250.jsonl")}
    from datasets import load_dataset
    train = load_dataset("openai/gsm8k", "main")["train"]
    pool = [(r["question"].strip(),
             r["answer"].split("####")[0].strip(),
             r["answer"].split("####")[-1].strip())
            for r in train if norm_q(r["question"]) not in dev]

    n_fs = int(len(rft) * args.fewshot_frac)
    rng.shuffle(rft)
    for r in rft[:n_fs]:
        shots = rng.sample(pool, rng.randint(1, 3))
        r["system"] = "\n\n".join(
            f"{q}\n\nReasoning:\n{reason}\n\nANSWER: {ans}"
            for q, reason, ans in shots)

    rng.shuffle(sft)
    rows = rft + sft[: args.replay]
    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_sys = sum(1 for r in rows if r.get("system"))
    print(json.dumps({"rft_rows": len(rft), "replay_rows": min(len(sft), args.replay),
                      "total": len(rows), "rows_with_fewshot_prefix": n_sys,
                      "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
