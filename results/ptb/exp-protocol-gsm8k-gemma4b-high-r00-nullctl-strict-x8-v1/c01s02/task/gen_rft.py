#!/usr/bin/env python3
"""Rejection-sampling data generation: sample solutions from the current policy,
keep those whose final answer matches the reference answer."""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict

PROMPT = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str):
    m = NUM_RE.findall(text)
    if not m:
        return None
    s = m[-1].replace(",", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v


def norm(x):
    if x is None:
        return None
    return round(x, 4)


def load_problems(args):
    probs = []
    if args.gsm8k_train:
        from datasets import load_dataset
        tr = load_dataset("openai/gsm8k", "main", split="train")
        for r in tr:
            ans = r["answer"].split("####")[-1].strip().replace(",", "")
            probs.append({"question": r["question"].strip(), "answer": ans, "src": "gsm8k_train"})
    if args.n_augmented > 0:
        import pyarrow.parquet as pq
        files = sorted(glob.glob(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
            "snapshots/*/data/train_1M-*.parquet"))
        seen = set()
        aug = []
        for f in files:
            d = pq.read_table(f, columns=["problem", "expected_answer", "problem_source"]).to_pydict()
            for p, a, s in zip(d["problem"], d["expected_answer"], d["problem_source"]):
                if s != "augmented_gsm8k":
                    continue
                p = p.strip()
                if p in seen:
                    continue
                seen.add(p)
                aug.append({"question": p, "answer": a.strip(), "src": "aug_gsm8k"})
            if len(aug) > args.n_augmented * 3:
                break
        random.Random(args.seed).shuffle(aug)
        probs += aug[: args.n_augmented]
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/rft.jsonl")
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-per-problem", type=int, default=4)
    ap.add_argument("--gsm8k-train", type=int, default=1)
    ap.add_argument("--n-augmented", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    probs = load_problems(args)
    print(f"{len(probs)} problems")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=2048,
        dtype="bfloat16",
        enforce_eager=False,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.n, temperature=args.temp, top_p=args.top_p,
        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed,
    )
    texts = [
        "<bos><start_of_turn>user\n" + PROMPT.format(prompt=p["question"])
        + "<end_of_turn>\n<start_of_turn>model\n"
        for p in probs
    ]
    outs = llm.generate(texts, sp)

    kept = []
    per_problem_correct = []
    for p, o in zip(probs, outs):
        gold = norm(last_number(p["answer"]))
        good, seen_txt = [], set()
        ncorrect = 0
        for c in o.outputs:
            t = c.text.strip()
            if not t or "ANSWER:" not in t:
                continue
            if norm(last_number(t)) != gold or gold is None:
                continue
            ncorrect += 1
            key = re.sub(r"\s+", " ", t)[:400]
            if key in seen_txt:
                continue
            seen_txt.add(key)
            good.append(t)
        per_problem_correct.append((p["question"], ncorrect, len(o.outputs), p["src"]))
        good.sort(key=len)
        for t in good[: args.max_per_problem]:
            kept.append({
                "prompt": PROMPT.format(prompt=p["question"]),
                "completion": t,
                "source": "rft_" + p["src"],
                "answer": p["answer"],
                "question": p["question"],
            })

    n_solved = sum(1 for _, c, _, _ in per_problem_correct if c > 0)
    print(f"kept {len(kept)} solutions; solved {n_solved}/{len(probs)} "
          f"({n_solved/len(probs):.1%}) at least once")
    bysrc = defaultdict(lambda: [0, 0, 0])
    for _, c, n, s in per_problem_correct:
        bysrc[s][0] += c
        bysrc[s][1] += n
        bysrc[s][2] += 1
    for s, (c, n, m) in bysrc.items():
        print(f"  {s}: pass-rate {c/n:.1%} over {m} problems")

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    if args.stats_out:
        with open(args.stats_out, "w") as f:
            for q, c, n, s in per_problem_correct:
                f.write(json.dumps({"question": q, "correct": c, "n": n, "src": s}) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
