#!/usr/bin/env python3
"""Add harness-style few-shot prefixes to SFT rows.

At eval time inspect_evals/gsm8k puts a 10-shot block in a system message,
which templates/gemma3.jinja folds into the first user turn as
    <fewshot block>\\n\\n<MATH_PROMPT_TEMPLATE with the question>
A model trained only on bare prompts keeps generating further question/answer
pairs instead of stopping (exp-02: 0.76 zero-shot vs 0.12 with the prefix).
This script rebuilds rows with the same prefix distribution so the model learns
to answer the last question and stop.

Few-shot examples are taken from the GSM8K TRAIN split (the same source the
harness draws them from) in the harness's own format, minus the held-out dev
questions.
"""
from __future__ import annotations

import argparse
import json
import random

from datasets import load_dataset


def build_shots(dev_q: set[str]) -> list[str]:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    shots = []
    for r in ds:
        q = r["question"].strip()
        if q in dev_q:
            continue
        body, _, ans = r["answer"].partition("####")
        shots.append(f"{q}\n\nReasoning:\n{body.strip()}\n\nANSWER: {ans.strip()}")
    return shots


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-src", required=True, help="jsonl for the few-shot-prefixed rows")
    ap.add_argument("--plain-src", required=True, help="jsonl for the zero-shot rows")
    ap.add_argument("--n-prefix", type=int, default=20000)
    ap.add_argument("--n-plain", type=int, default=15000)
    ap.add_argument("--min-shots", type=int, default=1)
    ap.add_argument("--max-shots", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    dev_q = {json.loads(l)["question"].strip() for l in open("data/dev_train300.jsonl")}
    shots = build_shots(dev_q)
    print("shot pool:", len(shots), flush=True)

    pref_rows = [json.loads(l) for l in open(args.prefix_src)]
    plain_rows = [json.loads(l) for l in open(args.plain_src)]
    rng.shuffle(pref_rows)
    rng.shuffle(plain_rows)
    pref_rows = pref_rows[: args.n_prefix]
    plain_rows = plain_rows[: args.n_plain]

    out = []
    for r in pref_rows:
        k = rng.randint(args.min_shots, args.max_shots)
        block = "\n\n".join(rng.sample(shots, k))
        r = dict(r)
        r["prompt"] = block + "\n\n" + r["prompt"]
        r["src"] = r.get("src", "?") + f"+fewshot{k}"
        out.append(r)
    for r in plain_rows:
        out.append(dict(r))
    rng.shuffle(out)

    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(out), "->", args.out, flush=True)

    with open(args.out.replace(".jsonl", "_for_decon.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")


if __name__ == "__main__":
    main()
