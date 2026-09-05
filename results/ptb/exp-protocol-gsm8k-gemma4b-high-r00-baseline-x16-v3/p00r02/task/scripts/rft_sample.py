#!/usr/bin/env python3
"""Rejection-sampling data generation: sample k solutions per question from a
checkpoint, keep the ones whose final answer is right.

Questions come from the *training* pools only (openai/gsm8k train,
OpenMathInstruct-2 problems already present in data/sft_v1.jsonl).  Prompts are
rendered with the same templates/gemma3.jinja the grader uses, and the answer is
extracted with the grader's own rule (last numeric whitespace token), so a
sample counts as correct exactly when the grader would score it correct.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402


def grader_answer(text: str) -> str | None:
    """inspect_ai match(location='end', numeric=True): last numeric token."""
    v = text.strip()
    v = re.sub(r"[,$]", "", v)
    for w in reversed(re.split(r"\s+", v)):
        w = w.strip(".;:!?%()[]{}\"'")
        if w.replace(".", "").replace("-", "").isnumeric():
            try:
                return format(float(w), ".5g")
            except ValueError:
                return None
    return None


def same(a: str, b: str) -> bool:
    try:
        return format(float(a), ".5g") == format(float(b), ".5g")
    except (TypeError, ValueError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with question/answer")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    qs = [json.loads(l) for l in open(args.questions)]
    if args.max_questions:
        random.Random(args.seed).shuffle(qs)
        qs = qs[: args.max_questions]
    print(f"[rft] {len(qs)} questions x k={args.k}", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=1536,
        seed=args.seed,
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=None,
    )
    prompts = [fmt.build_example(q["question"], "x", system=None)[0] for q in qs]
    outs = llm.generate(prompts, sp)

    kept, n_correct, n_total = [], 0, 0
    solved = 0
    for q, o in zip(qs, outs):
        good = []
        for c in o.outputs:
            n_total += 1
            text = c.text.strip()
            if not text:
                continue
            got = grader_answer(text)
            if got is None or not same(got, q["answer"]):
                continue
            if text.count("ANSWER: ") != 1 or not text.split("\n")[-1].startswith("ANSWER: "):
                continue
            n_correct += 1
            good.append(text)
        if good:
            solved += 1
        # keep more copies of the questions the model finds hard: an easy question
        # it already answers 8/8 adds nothing, a 1/8 question is the frontier
        rate = len(good) / max(1, len(o.outputs))
        cap = 1 if rate >= 0.75 else (2 if rate >= 0.375 else args.keep_per_question)
        # prefer the shortest correct solutions: less room for a lucky wrong chain
        good = sorted(set(good), key=len)[:cap]
        for g in good:
            kept.append({"question": q["question"], "solution": g, "answer": q["answer"],
                         "source": "rft:self"})

    print(f"[rft] pass@1 {n_correct / max(1, n_total):.3f}  "
          f"pass@{args.k} {solved / max(1, len(qs)):.3f}  kept {len(kept)}", flush=True)

    with open(args.out, "w") as f:
        for r in kept:
            prompt, target = fmt.build_example(r["question"], r["solution"], system=None)
            f.write(json.dumps({"prompt": prompt, "completion": target,
                                "question": r["question"], "answer": r["answer"],
                                "source": r["source"], "n_shot": 0}) + "\n")
    print("[rft] wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
