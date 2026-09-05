#!/usr/bin/env python3
"""Rejection-sampling data generation from the current checkpoint.

Samples k solutions per problem at a positive temperature, keeps the ones whose
final number equals the reference answer, dedups, and writes rows in exactly
the {prompt, completion, fewshot} schema train_sft.py reads - completions
already carrying <end_of_turn> so the stop_token preflight check sees the real
target.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

from local_eval import build_prompt, last_number, normalize
from train_sft import render


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True, help="jsonl of {question, gold}")
    ap.add_argument("--out", required=True)
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-keep-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-share", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.problems)]
    if args.limit:
        rows = rows[: args.limit]
    prompts = [render(build_prompt(r["question"], False), None) for r in rows]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        generation_config="vllm",
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    kept = []
    n_solved = 0
    per_problem_correct = []
    for r, o in zip(rows, outs):
        good = []
        n_ok = 0
        for cand in o.outputs:
            text = cand.text.strip()
            if cand.finish_reason != "stop":
                continue
            pred = last_number(text)
            if pred is None or normalize(pred) != normalize(r["gold"]):
                continue
            n_ok += 1
            # one answer marker, and it must be the tail of the completion
            if text.count("ANSWER:") != 1 or not re.search(r"ANSWER:\s*-?[\d.,]+$", text):
                continue
            good.append(text)
        per_problem_correct.append(n_ok)
        if not good:
            continue
        n_solved += 1
        # prefer shorter, non-degenerate solutions; keep distinct ones
        good = sorted(set(good), key=len)
        for text in good[: args.max_keep_per_problem]:
            kept.append({"question": r["question"], "completion": text + "<end_of_turn>"})

    rng.shuffle(kept)
    n_few = int(len(kept) * args.fewshot_share)
    few_idx = set(rng.sample(range(len(kept)), n_few)) if kept else set()
    with open(args.out, "w") as f:
        for i, row in enumerate(kept):
            fs = i in few_idx
            f.write(
                json.dumps(
                    {
                        "prompt": build_prompt(row["question"], fs),
                        "completion": row["completion"],
                        "fewshot": fs,
                    }
                )
                + "\n"
            )
    stats = {
        "problems": len(rows),
        "k": args.k,
        "problems_with_a_correct_sample": n_solved,
        "pass_at_k": n_solved / len(rows),
        "mean_correct_per_problem": sum(per_problem_correct) / (len(rows) * args.k),
        "rows_written": len(kept),
        "fewshot_rows": n_few,
        "out": args.out,
    }
    print(json.dumps(stats, indent=2))
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
