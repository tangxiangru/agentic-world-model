#!/usr/bin/env python3
"""Rejection-sampling data generation on the GSM8K TRAIN split.

Samples k solutions per train question from a fine-tuned checkpoint, keeps the ones
whose final number equals the gold answer, and writes them in the same
{"question", "target"} shape as data/sft_v1.jsonl. TRAIN split only - the test split is
never read here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

MATH_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text: str) -> str | None:
    ms = NUM.findall(text)
    if not ms:
        return None
    v = ms[-1].replace(",", "")
    if "." in v:
        v = v.rstrip("0").rstrip(".")
    return v or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-problem", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ds = load_dataset("openai/gsm8k", "main", split="train")
    if args.limit:
        ds = ds.select(range(args.limit))
    golds, questions = [], []
    for r in ds:
        questions.append(r["question"])
        golds.append(r["answer"].split("####")[-1].strip().replace(",", ""))

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open("/home/ben/task/templates/gemma3.jinja").read()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=2048,
        dtype="bfloat16",
        seed=args.seed,
        enable_prefix_caching=True,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    outs = llm.generate(prompts, sp)

    kept: dict[int, list[str]] = defaultdict(list)
    n_ok_per: dict[int, int] = defaultdict(int)
    n_correct = 0
    n_total = 0
    solved = 0
    for i, o in enumerate(outs):
        ok_here = 0
        for c in o.outputs:
            n_total += 1
            txt = c.text.strip()
            if not txt.rstrip().split("\n")[-1].startswith("ANSWER:"):
                continue
            pred = last_number(txt)
            if pred is None or pred != golds[i]:
                continue
            n_correct += 1
            ok_here += 1
            if txt not in kept[i] and len(kept[i]) < args.keep_per_problem:
                kept[i].append(txt)
        n_ok_per[i] = ok_here
        if ok_here:
            solved += 1

    rows = []
    for i, sols in kept.items():
        pr = n_ok_per[i] / args.k
        for s in sols:
            rows.append({"question": questions[i], "target": s, "source": "rft", "pass_rate": pr, "pid": i})
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats = {
        "n_problems": len(questions),
        "k": args.k,
        "samples": n_total,
        "correct_samples": n_correct,
        "pass_rate": n_correct / max(1, n_total),
        "problems_with_at_least_one": solved,
        "coverage": solved / max(1, len(questions)),
        "rows_written": len(rows),
    }
    print("RFTSTATS", json.dumps(stats))
    json.dump(stats, open(args.out + ".stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
