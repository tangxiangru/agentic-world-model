#!/usr/bin/env python3
"""Rejection-sampling data: sample k solutions per problem from a checkpoint,
keep the ones whose final answer is right.

Prompts are the grader's own prompt (scripts/evalfmt), so the samples are drawn
from the same distribution the model is graded in. Problems come from the
GSM8K TRAIN split and from OpenMathInstruct-2's gsm8k/augmented_gsm8k problems
(themselves derived from the train split). The GSM8K test split is never read.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

import pandas as pd
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalfmt as E  # noqa: E402
from build_sft_data import INT_ANS, OMI2_GLOB, last_number  # noqa: E402


def collect_problems(n_aug: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    probs: dict[str, str] = {}
    for r in load_dataset("openai/gsm8k", "main", split="train"):
        a = r["answer"].split("####")[-1].strip().replace(",", "")
        if INT_ANS.match(a):
            probs[r["question"].strip()] = a
    n_gold = len(probs)

    frames = []
    for f in sorted(glob.glob(OMI2_GLOB)):
        df = pd.read_parquet(f, columns=["problem", "expected_answer", "problem_source"])
        df = df[df["problem_source"] == "augmented_gsm8k"]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["expected_answer"].str.match(INT_ANS, na=False)]
    df = df.drop_duplicates(subset=["problem"])
    idx = list(range(len(df)))
    rng.shuffle(idx)
    added = 0
    for i in idx:
        if added >= n_aug:
            break
        p = df.iloc[i]["problem"].strip()
        if p in probs or not (40 <= len(p) <= 1200):
            continue
        probs[p] = df.iloc[i]["expected_answer"].strip()
        added += 1
    print(f"[problems] gsm8k train gold {n_gold} + augmented {added} = {len(probs)}")
    return [{"question": q, "answer": a} for q, a in probs.items()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--n-aug", type=int, default=20000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = E.chat_template()
    sysmsg = E.fewshot_system_message()

    problems = collect_problems(args.n_aug, args.seed)
    rng = random.Random(args.seed)
    prompts = []
    for p in problems:
        use = rng.random() < args.fewshot_frac
        prompts.append(
            tok.apply_chat_template(
                E.messages(p["question"], sysmsg if use else None),
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
        dtype="bfloat16",
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    outs = llm.generate(prompts, sp)

    kept, stats = [], []
    n_corr_total = 0
    for p, o in zip(problems, outs):
        texts = [c.text for c in o.outputs]
        good = []
        for t in texts:
            t = t.strip()
            if not t or "ANSWER:" not in t:
                continue
            if last_number(t) == p["answer"].lstrip("+"):
                good.append(t)
        n_corr_total += len(good)
        stats.append({"question": p["question"], "answer": p["answer"],
                      "k": len(texts), "n_correct": len(good)})
        if not good:
            continue
        good = sorted(set(good), key=len)          # prefer the shortest correct chains
        for t in good[: args.max_per_problem]:
            kept.append({"question": p["question"], "target": t + E.STOP_TOKEN,
                         "answer": p["answer"], "src": "rft_self"})

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    if args.stats_out:
        with open(args.stats_out, "w") as f:
            for s in stats:
                f.write(json.dumps(s) + "\n")
    solved = sum(1 for s in stats if s["n_correct"] > 0)
    print(f"[rft] problems {len(problems)}, solved>=1 {solved} ({solved/len(problems):.3f}), "
          f"pass@1 {n_corr_total/(len(problems)*args.k):.3f}, kept rows {len(kept)} -> {args.out}")


if __name__ == "__main__":
    main()
