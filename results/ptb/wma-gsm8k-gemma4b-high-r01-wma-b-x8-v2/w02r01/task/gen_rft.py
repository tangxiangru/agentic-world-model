#!/usr/bin/env python3
"""Rejection-sampling data generation: sample k solutions per gsm8k TRAIN question from
a checkpoint, keep the ones whose final answer matches the gold, dedupe, and write a
corpus in the same shape build_data.py produces.

Prompts are rendered with the same string the grader's templates/gemma3.jinja produces,
so the samples are drawn from the distribution the model is actually graded in.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

import pandas as pd

from build_data import (
    ANSWER_MARKER,
    GSM8K_TRAIN,
    MATH_PROMPT_TEMPLATE,
    STOP_TOKEN,
    clean_number,
)

BOS, EOT = "<bos>", "<end_of_turn>"
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def render_prompt(prompt: str) -> str:
    return f"{BOS}<start_of_turn>user\n{prompt.strip()}{EOT}\n<start_of_turn>model\n"


def final_number(text: str) -> str | None:
    """What match(numeric=True, location='end') would read: the last number in the text."""
    nums = _NUM.findall(text)
    if not nums:
        return None
    return clean_number(nums[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--limit-questions", type=int, default=-1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--easy-threshold", type=float, default=1.1,
                    help="a question is 'easy' when this fraction of its k samples are correct")
    ap.add_argument("--easy-cap", type=int, default=0,
                    help="rows kept for an easy question (0 disables the rebalance)")
    args = ap.parse_args()

    df = pd.read_parquet(GSM8K_TRAIN)
    items = []
    for q, a in zip(df["question"], df["answer"]):
        gold = clean_number(a.partition("####")[2])
        if gold is not None:
            items.append({"question": q.strip(), "gold": gold})
    if args.limit_questions > 0:
        items = items[: args.limit_questions]
    print(f"[rft] {len(items)} train questions, k={args.k}, T={args.temperature}")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=1024,
        seed=args.seed,
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    prompts = [render_prompt(MATH_PROMPT_TEMPLATE.format(prompt=it["question"])) for it in items]
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    kept, per_q_correct, n_correct, n_total = [], defaultdict(int), 0, 0
    for it, out in zip(items, outs):
        cands = []
        for o in out.outputs:
            n_total += 1
            text = o.text.strip()
            if text.count(ANSWER_MARKER) != 1:
                continue
            if final_number(text) != it["gold"]:
                continue
            n_correct += 1
            # trailing text after the answer line would give the grader a later number
            head, _, tail = text.rpartition(ANSWER_MARKER)
            text = head + ANSWER_MARKER + tail.strip().split("\n")[0].strip()
            if final_number(text) != it["gold"]:
                continue
            cands.append(text)
        per_q_correct[it["question"]] = len(cands)
        # dedupe identical chains, then keep a few distinct ones
        uniq = list(dict.fromkeys(cands))
        rng.shuffle(uniq)
        keep = args.keep_per_question
        if args.easy_cap > 0 and len(cands) >= args.easy_threshold * args.k:
            # a question the model already answers almost every time carries little
            # signal; capping it shifts the corpus mass onto the ones it gets wrong
            keep = args.easy_cap
        for text in uniq[:keep]:
            kept.append({
                "prompt": MATH_PROMPT_TEMPLATE.format(prompt=it["question"]),
                "completion": text + STOP_TOKEN,
                "question": it["question"],
                "answer": it["gold"],
            })

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    solved = sum(1 for v in per_q_correct.values() if v > 0)
    stats = {
        "questions": len(items),
        "samples_drawn": n_total,
        "samples_correct": n_correct,
        "pass_rate_per_sample": n_correct / max(1, n_total),
        "questions_with_at_least_one_correct": solved,
        "pass_at_k": solved / max(1, len(items)),
        "rows_written": len(kept),
    }
    print(json.dumps(stats, indent=2))
    if args.stats:
        json.dump(stats, open(args.stats, "w"), indent=2)


if __name__ == "__main__":
    main()
