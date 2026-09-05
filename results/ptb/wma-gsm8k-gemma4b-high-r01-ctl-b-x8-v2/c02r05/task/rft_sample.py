"""Rejection-sampling data generation: sample k solutions per question from the
current checkpoint, keep the ones whose ANSWER line matches the gold answer.

Self-generated, so nothing here is derived from the benchmark test set: the
questions come from the GSM8K TRAIN pool and OpenMathInstruct-2's gsm8k /
augmented_gsm8k problems, both of which are already in data/sft_v1.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

from prompt_fmt import END_OF_TURN, render_prompt

ANSWER_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")


def extract(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    if not m:
        return None
    v = m.group(1).replace(",", "")
    if v.endswith(".0"):
        v = v[:-2]
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with question/answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--n-questions", type=int, default=15000)
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    rows, seen = [], set()
    for line in open(args.questions):
        r = json.loads(line)
        if r["question"] in seen:
            continue
        seen.add(r["question"])
        rows.append(r)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n_questions]
    print(f"[rft] {len(rows)} unique questions", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=1024,
        dtype="bfloat16",
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
        seed=args.seed,
    )
    prompts = [render_prompt(r["question"]) for r in rows]
    outs = llm.generate(prompts, sp)

    kept, n_correct_any, per_q_correct = [], 0, []
    for r, o in zip(rows, outs):
        gold = str(r["answer"])
        good, texts = [], set()
        for c in o.outputs:
            t = c.text.strip()
            if extract(t) != gold:
                continue
            if t in texts:
                continue
            texts.add(t)
            good.append(t)
        per_q_correct.append(len(good))
        if good:
            n_correct_any += 1
        good.sort(key=len)  # prefer the most economical correct chains
        for t in good[: args.keep_per_question]:
            kept.append({"question": r["question"], "completion": t + END_OF_TURN,
                         "answer": gold, "src": "rft_self"})

    rng.shuffle(kept)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    stats = {
        "questions": len(rows),
        "k": args.k,
        "questions_with_a_correct_sample": n_correct_any,
        "pass_at_k": n_correct_any / max(1, len(rows)),
        "mean_correct_per_question": sum(per_q_correct) / max(1, len(per_q_correct)),
        "rows_written": len(kept),
        "out": args.out,
    }
    print(json.dumps(stats, indent=2), flush=True)
    if args.stats_out:
        json.dump(stats, open(args.stats_out, "w"), indent=2)


if __name__ == "__main__":
    main()
