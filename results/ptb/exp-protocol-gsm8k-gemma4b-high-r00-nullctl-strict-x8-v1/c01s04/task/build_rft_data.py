#!/usr/bin/env python3
"""Turn sampled generations into a rejection-sampling fine-tuning set."""
from __future__ import annotations

import argparse
import json
import random
import re

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

EQ_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def equation_sig(text):
    """Signature of the reasoning chain: the sequence of numbers used."""
    return tuple(EQ_RE.findall(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-q", type=int, default=2)
    ap.add_argument("--max-frac-correct", type=float, default=1.01,
                    help="drop questions the model already solves at >= this rate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = []
    stats = {"q": 0, "solved": 0, "kept": 0}
    for line in open(args.gen):
        r = json.loads(line)
        stats["q"] += 1
        good = [g for g in r["gens"] if g["correct"]]
        if not good:
            continue
        stats["solved"] += 1
        frac = len(good) / len(r["gens"])
        if frac >= args.max_frac_correct:
            continue
        # dedup by numeric signature, prefer well-formed and reasonably short
        seen = set()
        cand = []
        for g in good:
            t = g["text"].strip()
            if not t.rstrip().endswith(str(r["answer"])):
                # must end with the ANSWER line
                if "ANSWER:" not in t:
                    continue
            if len(t) < 20 or len(t) > 3000:
                continue
            sig = equation_sig(t)
            if sig in seen:
                continue
            seen.add(sig)
            cand.append(t)
        if not cand:
            continue
        cand.sort(key=len)
        rng.shuffle(cand)
        for t in cand[: args.max_per_q]:
            out.append({
                "prompt": MATH_PROMPT_TEMPLATE.format(prompt=r["question"].strip()),
                "completion": t,
                "question": r["question"].strip(),
                "answer": r["answer"],
            })
            stats["kept"] += 1

    rng.shuffle(out)
    with open(args.out, "w") as f:
        for d in out:
            f.write(json.dumps(d) + "\n")
    print(stats, "->", args.out)


if __name__ == "__main__":
    main()
