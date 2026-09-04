#!/usr/bin/env python3
"""Render the rejection-sampled rows into prompt/completion pairs and mix them with a
slice of the original SFT corpus, so the second stage does not narrow onto its own
output distribution."""
from __future__ import annotations
import argparse, json, random
from collections import Counter
from transformers import AutoTokenizer

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", required=True)
    ap.add_argument("--sft", default="data/sft_train.jsonl")
    ap.add_argument("--n-sft", type=int, default=20000)
    ap.add_argument("--sft-src", default="",
                    help="comma-separated src prefixes to keep from the SFT corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-out", default="data/rft_check.jsonl")
    ap.add_argument("--kshot-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    tok = AutoTokenizer.from_pretrained(SNAP)
    tok.chat_template = open("templates/gemma3.jinja").read()
    sysmsg = open("data/eval_system_message.txt").read()

    rft = [json.loads(l) for l in open(args.rft)]
    rng.shuffle(rft)
    n_kshot = int(len(rft) * args.kshot_frac)
    rows = []
    for i, r in enumerate(rft):
        user = MATH_PROMPT_TEMPLATE.replace("{prompt}", r["question"])
        kshot = i < n_kshot
        msgs = ([{"role": "system", "content": sysmsg}] if kshot else []) + \
               [{"role": "user", "content": user}]
        rows.append({"prompt": tok.apply_chat_template(msgs, tokenize=False,
                                                       add_generation_prompt=True),
                     "completion": r["completion"], "src": "rft_self",
                     "kshot": int(kshot), "question": r["question"]})

    sft = [json.loads(l) for l in open(args.sft)]
    if args.sft_src:
        keep = tuple(args.sft_src.split(","))
        sft = [r for r in sft if r["src"].startswith(keep)]
        print(f"SFT pool filtered to {keep}: {len(sft)} rows available")
    rng.shuffle(sft)
    rows += sft[: args.n_sft]
    rng.shuffle(rows)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(args.check_out, "w") as f:
        for r in rows:
            if r["src"] == "rft_self":
                f.write(json.dumps({"text": r["question"] + "\n" + r["completion"]}) + "\n")
    print(Counter(r["src"] for r in rows), "total", len(rows))


if __name__ == "__main__":
    main()
