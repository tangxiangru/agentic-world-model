#!/usr/bin/env python3
"""Rejection-sampling data generation: sample k solutions per training question
from a checkpoint with vLLM, keep the ones whose final number equals gold.

Prompts are rendered with templates/gemma3.jinja, exactly as the grader does.
Questions come from the GSM8K *train* split and OMI-2 augmented problems only.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
MATH_PROMPT_TEMPLATE = (
    'Solve the following math problem step by step. The last line of your response should be of '
    'the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
    "{prompt}\n\n"
    'Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" '
    "(without quotes) where $ANSWER is the answer to the problem, and you do not need to use a "
    "\\boxed command.\n\nReasoning:"
)


def last_number(text: str) -> str | None:
    nums = NUM.findall(text.strip())
    if not nums:
        return None
    s = nums[-1].replace(",", "")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with {question, answer}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-keep-per-question", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--template", default="/home/ben/task/templates/gemma3.jinja")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open(args.template).read()

    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        rows = rows[: args.limit]
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=r["question"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in rows
    ]
    print(f"{len(prompts)} prompts x n={args.n}", flush=True)

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=1536,
        dtype="bfloat16",
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[tok.convert_tokens_to_ids("<end_of_turn>")],
        seed=args.seed,
    )
    outs = llm.generate(prompts, sp)

    kept, per_q = [], defaultdict(int)
    n_samples = n_correct = 0
    solved = 0
    for r, o in zip(rows, outs):
        gold = str(r["answer"]).strip()
        cands = []
        for c in o.outputs:
            n_samples += 1
            text = c.text.strip()
            if last_number(text) == gold and text.count("ANSWER:") == 1:
                n_correct += 1
                cands.append(text)
        if cands:
            solved += 1
        seen = set()
        rng.shuffle(cands)
        for text in cands:
            if per_q[r["question"]] >= args.max_keep_per_question:
                break
            if text in seen:
                continue
            seen.add(text)
            per_q[r["question"]] += 1
            kept.append(
                {
                    "question": r["question"],
                    "completion": text + "<end_of_turn>",
                    "source": "rft_self",
                    "nshot": 0,
                }
            )

    rng.shuffle(kept)
    with open(args.out, "w") as fh:
        for i, rec in enumerate(kept):
            rec["id"] = f"rft-{i:07d}"
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "questions": len(rows),
        "samples": n_samples,
        "correct_samples": n_correct,
        "sample_accuracy": n_correct / max(1, n_samples),
        "questions_solved_at_least_once": solved,
        "pass_at_n": solved / max(1, len(rows)),
        "kept": len(kept),
    }
    print(json.dumps(stats, indent=2))
    if args.stats_out:
        json.dump(stats, open(args.stats_out, "w"), indent=2)


if __name__ == "__main__":
    main()
