#!/usr/bin/env python3
"""Rejection-sampling data generation: sample k solutions per training question
from a checkpoint, keep only those whose final ANSWER matches the known answer.

Questions come from openai/gsm8k split=train and OpenMathInstruct-2's
gsm8k-family problems (both train-derived); no benchmark test item is involved.
Prompts are the byte-exact grader render, so the samples are on-policy for the
distribution the grader will see.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict

from build_data import (MATH_PROMPT_TEMPLATE, gsm8k_train_rows, omi_rows,
                        render_prompt)


def final_answer(text: str):
    m = re.findall(r"ANSWER:\s*([^\n]+)", text)
    if not m:
        return None
    v = m[-1].strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    try:
        return float(v)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-questions", type=int, default=20000)
    ap.add_argument("--gsm8k-share", type=float, default=0.4)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    gsm = list(gsm8k_train_rows())
    n_gsm = min(len(gsm), int(args.n_questions * args.gsm8k_share))
    rng.shuffle(gsm)
    qs = gsm[:n_gsm]

    seen = set(hashlib.md5(q["q"].encode()).hexdigest() for q in qs)
    omi = []
    for r in omi_rows(1):
        h = hashlib.md5(r["q"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        omi.append(r)
    rng.shuffle(omi)
    qs += omi[: args.n_questions - n_gsm]
    rng.shuffle(qs)
    print(f"[rft] {len(qs)} questions ({n_gsm} gsm8k train, {len(qs)-n_gsm} OMI-augmented)")

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=1536, enforce_eager=False, seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106])

    prompts = [render_prompt(None, MATH_PROMPT_TEMPLATE.format(prompt=q["q"])) for q in qs]
    outs = llm.generate(prompts, sp)

    kept = defaultdict(list)
    n_corr = n_tot = 0
    solved = 0
    for q, o in zip(qs, outs):
        gold = float(q["ans"])
        good = []
        for c in o.outputs:
            n_tot += 1
            t = c.text.strip()
            a = final_answer(t)
            if a is not None and abs(a - gold) < 1e-6:
                n_corr += 1
                good.append(t)
        if good:
            solved += 1
        # dedupe near-identical samples, prefer the shortest correct ones
        uniq, seen_sig = [], set()
        for t in sorted(good, key=len):
            sig = re.sub(r"\s+", " ", t)[:200]
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            uniq.append(t)
        kept[q["q"]] = (uniq[: args.keep_per_question], q)

    n_rows = 0
    with open(args.out, "w") as f:
        for _, (sols, q) in kept.items():
            for t in sols:
                if not t.endswith(f"ANSWER: {q['ans']}"):
                    # normalise the trailing line so the graded number is exact
                    t = re.sub(r"ANSWER:\s*[^\n]+\s*$", f"ANSWER: {q['ans']}", t)
                f.write(json.dumps({
                    "prompt": render_prompt(None, MATH_PROMPT_TEMPLATE.format(prompt=q["q"])),
                    "completion": t + "<end_of_turn>",
                    "source": "rft:" + q["src"],
                    "answer": q["ans"],
                    "question": q["q"],
                }) + "\n")
                n_rows += 1
    print(f"[rft] sample-level pass rate {n_corr}/{n_tot} = {n_corr/max(1,n_tot):.3f}")
    print(f"[rft] questions solved at least once: {solved}/{len(qs)} = {solved/len(qs):.3f}")
    print(f"[rft] wrote {n_rows} rows to {args.out}")


if __name__ == "__main__":
    main()
