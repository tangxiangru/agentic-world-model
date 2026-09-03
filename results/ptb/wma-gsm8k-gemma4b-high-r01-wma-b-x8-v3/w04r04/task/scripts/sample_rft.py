#!/usr/bin/env python3
"""Rejection-sampling data generation from an SFT checkpoint.

Samples k solutions per problem with vLLM using the *grader's* chat template, keeps the
ones whose final 'ANSWER: N' matches gold, and writes them in the same jsonl schema
scripts/train_sft.py consumes. Problems come from the GSM8K *train* split and from
OpenMathInstruct-2's gsm8k-derived problems; the GSM8K test split is never read.
"""
from __future__ import annotations

import argparse
import json
import random
import re

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUMERIC = re.compile(r"^-?\d[\d,]*(\.\d+)?$")
ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")


def norm(a: str):
    try:
        return round(float(str(a).strip().replace(",", "").replace("$", "")), 4)
    except ValueError:
        return None


def extract(text: str):
    m = ANS_RE.findall(text)
    return norm(m[-1]) if m else None


def load_problems(args) -> list[dict]:
    probs = []
    d = load_dataset("openai/gsm8k", "main", split="train")
    for r in d:
        gold = norm(r["answer"].rsplit("####", 1)[-1])
        if gold is not None:
            probs.append({"q": r["question"].strip(), "gold": gold, "src": "gsm8k_train"})
    if args.n_aug:
        aug = load_from_disk(args.omi2_dir)
        seen = set()
        pool = []
        for r in aug:
            q = r["problem"].strip()
            if q in seen or not NUMERIC.match(str(r["expected_answer"]).strip()):
                continue
            seen.add(q)
            g = norm(r["expected_answer"])
            if g is not None:
                pool.append({"q": q, "gold": g, "src": "augmented_gsm8k"})
        random.Random(args.seed).shuffle(pool)
        probs += pool[: args.n_aug]
    return probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--n-aug", type=int, default=40000)
    ap.add_argument("--omi2-dir", default="/home/ben/task/data/omi2_gsm_1M")
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    template = open(TEMPLATE).read()

    probs = load_problems(args)
    if args.limit:
        probs = probs[: args.limit]
    print(f"problems: {len(probs)}")

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=p["q"])}],
            chat_template=template, tokenize=False, add_generation_prompt=True,
        )
        for p in probs
    ]

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=2048,
              dtype="bfloat16", seed=args.seed, enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    n_kept = n_solved = 0
    per_problem_correct = []
    with open(args.out, "w") as f:
        for p, o in zip(probs, outs):
            cands, seen = [], set()
            for c in o.outputs:
                t = c.text.strip()
                if c.finish_reason != "stop" or extract(t) != p["gold"]:
                    continue
                if t.count("ANSWER:") != 1 or t in seen:
                    continue
                seen.add(t)
                cands.append(t)
            per_problem_correct.append(len(cands))
            if not cands:
                continue
            n_solved += 1
            rng.shuffle(cands)
            for t in cands[: args.keep]:
                f.write(json.dumps({
                    "system": None,
                    "prompt": MATH_PROMPT_TEMPLATE.format(prompt=p["q"]),
                    "completion": t + "<end_of_turn>",
                    "answer": str(p["gold"]),
                    "src": "rft:" + p["src"],
                }) + "\n")
                n_kept += 1
    print(f"solved {n_solved}/{len(probs)} problems; wrote {n_kept} rows to {args.out}")
    if args.stats_out:
        json.dump({
            "n_problems": len(probs), "n_solved": n_solved, "n_rows": n_kept, "k": args.k,
            "pass_at_1_estimate": sum(per_problem_correct) / (args.k * len(probs)),
            "solve_rate": n_solved / len(probs),
        }, open(args.stats_out, "w"), indent=2)


if __name__ == "__main__":
    main()
