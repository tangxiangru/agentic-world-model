#!/usr/bin/env python3
"""Build SFT rows whose prompt carries a k-shot prefix in the harness's own shape.

inspect_evals/gsm8k puts 10 worked GSM8K *train* examples in a system message,
formatted as "<question>\n\nReasoning:\n<gsm8k solution>\n\nANSWER: <n>" joined by
blank lines; templates/gemma3.jinja folds that system message into the first user turn.
Training rows built here reproduce that shape with k drawn in [--kmin, --kmax], so the
model learns to end its turn after the FIRST answer even when in-context examples show
one problem following another.

Few-shot examples come from the GSM8K train split only. The test split is never read.
"""
from __future__ import annotations

import argparse
import json
import random

from datasets import load_dataset

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def fewshot_block(rec):
    """Exactly inspect_evals.gsm8k.sample_to_fewshot."""
    q = rec["question"]
    reasoning, target = rec["answer"].split("####")
    return f"{q}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {target.strip()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/sft_gsm8k.jsonl")
    ap.add_argument("--exclude", default="data/sft_gsm8k_1pp.jsonl",
                    help="rows already used in the previous stage; their completions are skipped")
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--zero-shot-frac", type=float, default=0.2)
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data/sft_fewshot.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    shots = [fewshot_block(r) for r in load_dataset("openai/gsm8k", "main", split="train")]
    print("fewshot pool:", len(shots))

    used = set()
    for path in filter(None, args.exclude.split(",")):
        for line in open(path):
            used.add(json.loads(line)["completion"])
    pool = []
    for line in open(args.pool):
        d = json.loads(line)
        if d["completion"] in used:
            continue
        pool.append(d)
    print("candidate rows (unseen solutions):", len(pool))
    rng.shuffle(pool)
    pool = pool[: args.n]

    rows = []
    for d in pool:
        base = PROMPT_TEMPLATE.format(prompt=d["question"])
        if rng.random() < args.zero_shot_frac:
            prompt = base
            k = 0
        else:
            k = rng.randint(args.kmin, args.kmax)
            prefix = "\n\n".join(rng.sample(shots, k))
            # gemma3.jinja folds a system message in as "<system>\n\n" before the user text
            prompt = prefix + "\n\n" + base
        rows.append({
            "prompt": prompt,
            "completion": d["completion"],
            "question": d["question"],
            "answer": d["answer"],
            "source": f"fewshot_k{k}",
        })
    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    lens = sorted(len(r["prompt"]) + len(r["completion"]) for r in rows)
    print("wrote", len(rows), "rows;",
          "chars p50/p95/max:", lens[len(lens) // 2], lens[int(.95 * len(lens))], lens[-1])


if __name__ == "__main__":
    main()
