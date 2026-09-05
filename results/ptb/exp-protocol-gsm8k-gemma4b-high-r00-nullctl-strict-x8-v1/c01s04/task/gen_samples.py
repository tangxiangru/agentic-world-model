#!/usr/bin/env python3
"""Sample solutions from a fine-tuned checkpoint with vLLM for rejection-sampling FT."""
from __future__ import annotations

import argparse
import json
import os
import random
import re

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def norm(s):
    s = str(s).strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else ("%f" % f).rstrip("0").rstrip(".")
    except Exception:
        return s


def extract(text):
    m = re.findall(r"ANSWER:\s*([^\n]*)", text)
    if m:
        nums = NUM_RE.findall(m[-1])
        if nums:
            return norm(nums[-1])
    nums = NUM_RE.findall(text)
    return norm(nums[-1]) if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with question/answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} questions x {args.n}")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=2048,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    prompts = [
        "<bos><start_of_turn>user\n" + MATH_PROMPT_TEMPLATE.format(prompt=r["question"].strip())
        + "<end_of_turn>\n<start_of_turn>model\n"
        for r in rows
    ]
    sp = SamplingParams(
        n=args.n, temperature=args.temp, top_p=args.top_p, top_k=64,
        max_tokens=args.max_tokens, stop_token_ids=[1, 106],
    )
    outs = llm.generate(prompts, sp)

    n_ok = 0
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            gold = norm(r["answer"])
            gens = []
            for c in o.outputs:
                t = c.text
                gens.append({"text": t, "pred": extract(t), "correct": extract(t) == gold})
            n_ok += any(g["correct"] for g in gens)
            f.write(json.dumps({"question": r["question"], "answer": gold, "gens": gens}) + "\n")
    print(f"solvable (pass@{args.n}): {n_ok}/{len(rows)} = {n_ok/len(rows):.3f}")


if __name__ == "__main__":
    main()
