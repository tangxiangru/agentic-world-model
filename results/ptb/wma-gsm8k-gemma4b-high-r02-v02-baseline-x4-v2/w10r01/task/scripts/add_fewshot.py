#!/usr/bin/env python3
"""Re-render a slice of an SFT file with a harness-style k-shot system prefix.

The harness always grades with a 10-shot system message whose exemplars are raw
GSM8K *train* solutions - calculator annotations and all - joined by blank
lines and folded into the first user turn by gemma3.jinja. Training is 0-shot,
so this exists to make the model robust to that prefix if the watch40 probe
shows it is not.

Exemplars are drawn from train_pool.jsonl (the same held-out-free pool the SFT
data comes from); the benchmark test split is not touched.
"""
from __future__ import annotations

import argparse
import json
import random

from common import TASK_DIR, render_prompt


def load_pool():
    rows = [json.loads(l) for l in (TASK_DIR / "data" / "train_pool.jsonl").open()]
    out = []
    for r in rows:
        reasoning = r["answer"].split("####")[0].strip()
        out.append(f"{r['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {r['gold']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--shots", type=int, nargs="+", default=[2, 4, 8, 10])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = load_pool()
    rows = [json.loads(l) for l in open(args.src)]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    with open(args.out, "w") as f:
        for r in rows[: args.n]:
            k = rng.choice(args.shots)
            system = "\n\n".join(rng.sample(pool, k))
            f.write(
                json.dumps(
                    {
                        "prompt": render_prompt(r["question"], system),
                        "target": r["target"],
                        "src": f"fewshot{k}:" + r["src"],
                        "question": r["question"],
                        "answer": r["answer"],
                    }
                )
                + "\n"
            )
    print(f"{args.out}: {min(args.n, len(rows))} rows with {args.shots}-shot prefixes")


if __name__ == "__main__":
    main()
