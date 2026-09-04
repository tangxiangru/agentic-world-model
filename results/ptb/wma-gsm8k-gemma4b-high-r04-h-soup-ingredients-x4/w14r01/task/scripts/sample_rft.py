#!/usr/bin/env python3
"""Rejection sampling: draw k solutions per GSM8K *train* question from a
checkpoint, keep the ones whose final 'ANSWER: N' equals the gold answer.

Prompts are rendered with the same string the grader produces, so the samples
are in-distribution for the eval.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    ANSWER_MARKER,
    STOP_TOKEN,
    normalize_number,
    render_prompt,
    strip_calc_annotations,
    user_text,
)


def extract_answer(text: str):
    m = re.findall(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if not m:
        return None
    return normalize_number(m[-1])


def canon(sol: str) -> str:
    """Signature for near-duplicate removal: the sequence of numbers used."""
    return "|".join(re.findall(r"-?\d+(?:\.\d+)?", sol))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    ds = load_dataset("openai/gsm8k", "main")["train"]
    items = []
    for r in ds:
        body, final = r["answer"].rsplit("####", 1)
        n = normalize_number(final)
        if n is None:
            continue
        items.append(
            {
                "question": r["question"].strip(),
                "gold": n,
                "ref": strip_calc_annotations(body).strip(),
            }
        )
    if args.limit:
        items = items[: args.limit]
    print(f"[rft] {len(items)} train questions", flush=True)

    prompts = [render_prompt(user_text(it["question"])) for it in items]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=1536,
        seed=args.seed,
        dtype="bfloat16",
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
    )
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    kept = 0
    solved = 0
    n_samples = 0
    n_correct = 0
    per_q = defaultdict(list)
    for it, o in zip(items, outs):
        cands = []
        for c in o.outputs:
            txt = c.text.strip()
            n_samples += 1
            a = extract_answer(txt)
            if a is None or a != it["gold"]:
                continue
            n_correct += 1
            # normalise the tail so exactly one marker survives and it is last
            idx = txt.rfind(ANSWER_MARKER)
            body = txt[:idx].strip()
            if not body or body.count(ANSWER_MARKER) > 0:
                continue
            if len(body) < 10:
                continue
            cands.append(body + "\n\n" + ANSWER_MARKER + it["gold"] + STOP_TOKEN)
        if not cands:
            continue
        solved += 1
        seen = set()
        uniq = []
        rng.shuffle(cands)
        for c in cands:
            s = canon(c)
            if s in seen:
                continue
            seen.add(s)
            uniq.append(c)
        per_q[it["question"]] = (it, uniq[: args.max_per_question])
        kept += len(uniq[: args.max_per_question])

    with open(args.out, "w") as f:
        for q, (it, sols) in per_q.items():
            for s in sols:
                f.write(
                    json.dumps(
                        {
                            "prompt": user_text(it["question"]),
                            "completion": s,
                            "question": it["question"],
                            "answer": it["gold"],
                            "src": "rft_self",
                            "fewshot": False,
                        }
                    )
                    + "\n"
                )
    stats = {
        "questions": len(items),
        "samples": n_samples,
        "sample_accuracy": n_correct / max(1, n_samples),
        "questions_solved_at_least_once": solved,
        "coverage": solved / max(1, len(items)),
        "rows_written": kept,
        "model": args.model,
        "k": args.k,
        "temperature": args.temperature,
    }
    print(json.dumps(stats, indent=2), flush=True)
    if args.stats:
        with open(args.stats, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
