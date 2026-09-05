#!/usr/bin/env python3
"""Rejection-sampling data generation from a fine-tuned checkpoint.

Samples k solutions per training question with vLLM, keeps the ones whose
'ANSWER: n' line matches the gold answer, dedupes near-identical chains, and
writes rows in the same schema as build_data.py (messages + completion).
Questions come from GSM8K TRAIN and OpenMathInstruct-2 problems only; the dev
holdout and the official test split are never touched.
"""
from __future__ import annotations

import argparse
import json
import random
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

TEMPLATE = "/home/ben/task/templates/gemma3.jinja"
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANS = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")


def norm(x: str) -> str:
    x = x.replace(",", "").rstrip(".")
    return x.split(".")[0] if x.endswith(".0") or "." not in x else x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with question/answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-questions", type=int, default=15000)
    ap.add_argument("--max-keep", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    qs = [json.loads(l) for l in open(args.questions)]
    rng.shuffle(qs)
    qs = qs[: args.n_questions]
    print(f"{len(qs)} questions x k={args.k}")

    tok = AutoTokenizer.from_pretrained(args.model)
    template = open(TEMPLATE).read()
    prompts = []
    for q in qs:
        msgs = [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q["question"])}]
        prompts.append(tok.apply_chat_template(msgs, chat_template=template,
                                               tokenize=False, add_generation_prompt=True))

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=2048, enable_prefix_caching=True, seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    n_kept = n_solved = 0
    with open(args.out, "w") as f:
        for q, o in zip(qs, outs):
            gold = norm(str(q["answer"] if "answer" in q else q["gold"]))
            good = []
            seen = set()
            for c in o.outputs:
                text = c.text.strip()
                m = ANS.search(text)
                if not m or norm(m.group(1)) != gold:
                    continue
                if text.count("ANSWER:") != 1 or not text.rstrip().endswith(m.group(0).strip()):
                    continue
                key = re.sub(r"\s+", " ", text)[:120]
                if key in seen:
                    continue
                seen.add(key)
                good.append(text)
            if good:
                n_solved += 1
            good.sort(key=len)
            for text in good[: args.max_keep]:
                msgs = [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q["question"])},
                        {"role": "assistant", "content": text}]
                f.write(json.dumps({"messages": msgs, "answer": gold, "src": "rft_self",
                                    "completion": text + "<end_of_turn>"}) + "\n")
                n_kept += 1
    print(f"solved at least once: {n_solved}/{len(qs)}; rows written: {n_kept} -> {args.out}")


if __name__ == "__main__":
    main()
