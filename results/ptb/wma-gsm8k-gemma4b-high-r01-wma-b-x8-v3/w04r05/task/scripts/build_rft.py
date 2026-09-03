#!/usr/bin/env python3
"""Turn scripts/vllm_gen.py samples into a rejection-sampling-FT corpus.

Keeps only samples the grader would score correct, caps per problem, and
prefers solutions that are short and distinct. Problems the model already
solves 4/4 are down-weighted to one solution and problems it solves 1-2/4
keep two, so the corpus leans toward what the model is still shaky on
instead of re-teaching what it already knows.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import render_completion, render_prompt, sample_to_fewshot  # noqa: E402


def norm(text: str) -> str:
    """Cheap dedup key: the sequence of numbers the solution computes."""
    return "|".join(re.findall(r"-?\d+\.?\d*", text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default="analysis/rft_stats.json")
    ap.add_argument("--anchor", default=None, help="sft jsonl to mix in")
    ap.add_argument("--n-anchor", type=int, default=20000)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    dev = {json.loads(l)["question"].strip() for l in open("data/dev_train500.jsonl")}
    pool = []
    for r in gsm:
        q = r["question"].strip()
        if q in dev:
            continue
        reasoning, ans = r["answer"].split("####")
        pool.append({"question": q, "reasoning": reasoning.strip(), "answer": ans.strip()})
    fewshot_pool = pool[:2000]

    def prefix():
        if rng.random() >= args.fewshot_frac:
            return None
        k = rng.randint(2, 8)
        return "\n\n".join(
            sample_to_fewshot(s["question"], s["reasoning"], s["answer"])
            for s in rng.sample(fewshot_pool, k)
        )

    rows = []
    hist = Counter()
    n_prob = 0
    n_solved = 0
    for path in args.samples:
      for line in open(path):
          r = json.loads(line)
          n_prob += 1
          good = [g for g in r["gens"] if g["correct"] and g["finish"] == "stop"]
          hist[len(good)] += 1
          if not good:
              continue
          n_solved += 1
          # keep more solutions where the model is unreliable
          keep = 1 if len(good) >= 4 else 2
          good.sort(key=lambda g: g["ntok"])
          chosen, seen = [], set()
          for g in good:
              k = norm(g["text"])
              if k in seen:
                  continue
              seen.add(k)
              chosen.append(g)
              if len(chosen) >= keep:
                  break
          for g in chosen:
              rows.append(
                  {
                      "prompt": render_prompt(r["question"], prefix()),
                      "completion": render_completion(g["text"]),
                      "answer": r["gold"],
                      "src": f"rft_pass{len(good)}of{len(r['gens'])}",
                  }
              )

    n_rft = len(rows)
    if args.anchor:
        anchor = [json.loads(l) for l in open(args.anchor)]
        rng.shuffle(anchor)
        rows.extend(anchor[: args.n_anchor])

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    stats = {
        "problems_sampled": n_prob,
        "problems_with_a_correct_sample": n_solved,
        "pass_at_k": n_solved / max(1, n_prob),
        "pass_count_histogram": {str(k): v for k, v in sorted(hist.items())},
        "rft_rows": n_rft,
        "anchor_rows": len(rows) - n_rft,
        "total_rows": len(rows),
        "out": args.out,
    }
    print(json.dumps(stats, indent=2))
    json.dump(stats, open(args.stats_out, "w"), indent=2)


if __name__ == "__main__":
    main()
