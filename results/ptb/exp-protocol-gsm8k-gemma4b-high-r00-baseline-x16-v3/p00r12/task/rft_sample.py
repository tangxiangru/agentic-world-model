#!/usr/bin/env python3
"""Rejection-sampling fine-tuning data: sample k solutions per GSM8K-train
problem from a checkpoint, keep the ones whose final ANSWER matches gold.

Uses the same chat template the grader uses, so the samples are on-policy for
the graded prompt shape.  The GSM8K *test* split is never touched.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

from build_data import (
    MATH_PROMPT_TEMPLATE,
    CALC_RE,
    STOP,
    fewshot_block,
    normalize_answer,
)

ANS_RE = re.compile(r"ANSWER:\s*(\$?-?[\d,]*\.?\d+)\s*$")


def extract(text: str) -> str | None:
    m = ANS_RE.search(text.strip())
    if not m:
        return None
    return normalize_answer(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--limit-problems", type=int, default=0)
    ap.add_argument("--extra-problems", type=int, default=0,
                    help="additionally sample this many augmented_gsm8k problems")
    ap.add_argument("--fewshot-frac", type=float, default=0.25)
    ap.add_argument("--max-shots", type=int, default=4)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    template = open("templates/gemma3.jinja").read()
    tok = AutoTokenizer.from_pretrained(args.model)

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    shot_pool, problems = [], []
    for r in gsm:
        q = r["question"].strip()
        body, _, tgt = r["answer"].rpartition("####")
        shot_pool.append((q, CALC_RE.sub("", body).strip(), tgt.strip()))
        a = normalize_answer(tgt)
        if a is not None:
            problems.append((q, a, "gsm8k_train"))
    if args.limit_problems:
        problems = problems[: args.limit_problems]

    if args.extra_problems:
        omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
        omi = omi.filter(lambda x: x["problem_source"] == "augmented_gsm8k", num_proc=8)
        idx = list(range(len(omi)))
        rng.shuffle(idx)
        seen = set()
        for i in idx:
            if len(seen) >= args.extra_problems:
                break
            r = omi[i]
            p = r["problem"].strip()
            a = normalize_answer(r["expected_answer"])
            if a is None or p in seen:
                continue
            seen.add(p)
            problems.append((p, a, "augmented_gsm8k"))

    print(f"problems: {len(problems)}  k={args.k}")

    prompts, meta = [], []
    for q, gold, src in problems:
        system = None
        if rng.random() < args.fewshot_frac:
            system = fewshot_block(rng.sample(shot_pool, rng.randint(1, args.max_shots)))
        user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        text = tok.apply_chat_template(
            msgs, chat_template=template, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
        meta.append((q, gold, src, system, user))

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=3072,
        dtype="bfloat16",
        enforce_eager=False,
    )
    # no seed: with n>1 a fixed per-request seed makes vLLM return near-identical
    # samples, which collapses the rejection-sampling yield (seen in the smoke run)
    sp = SamplingParams(
        n=args.k, temperature=args.temp, top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    outs = llm.generate(prompts, sp)

    kept, n_correct, n_total = [], 0, 0
    per_q_correct = []
    for o, (q, gold, src, system, user) in zip(outs, meta):
        texts = [c.text for c in o.outputs]
        good = []
        for t in texts:
            n_total += 1
            if extract(t) == gold:
                n_correct += 1
                body = t.strip()
                if body.count("ANSWER:") != 1:
                    continue
                good.append(body)
        per_q_correct.append(len(good))
        # prefer distinct, shortest-first chains
        good = sorted(set(good), key=len)[: args.keep_per_problem]
        for g in good:
            kept.append({
                "question": q, "answer": gold, "source": f"rft:{src}",
                "system": system, "user": user, "nshot": 0 if system is None else 1,
                "target": g + STOP,
            })

    rng.shuffle(kept)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    check = args.out.replace(".jsonl", "_check.jsonl")
    with open(check, "w") as f:
        for r in kept:
            f.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")

    solved = sum(1 for c in per_q_correct if c > 0)
    print(f"samples: {n_total} correct: {n_correct} ({n_correct/n_total:.3f})")
    print(f"problems with >=1 correct: {solved}/{len(problems)} ({solved/len(problems):.3f})")
    print(f"kept rows: {len(kept)} -> {args.out}")


if __name__ == "__main__":
    main()
