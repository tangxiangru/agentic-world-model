#!/usr/bin/env python3
"""Score a checkpoint on the held-out gsm8k-TRAIN probe set (data/probe200.jsonl).

This is NOT the benchmark: the items come from the train split and are excluded
from every training mixture, so it can be used freely for iteration.  The prompt
rendering and the grading rule are copied from the real grader (render.py and
inspect_ai.scorer._common.match_str with numeric=True, location='end').
"""
from __future__ import annotations

import argparse
import json
import os

from inspect_ai.scorer._common import match_str

import render


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/probe200.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fewshot", type=int, default=10, help="0 = zero-shot prompt")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--n", type=int, default=1, help="samples per item (majority vote if >1)")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]

    system = render.fewshot_system_message(args.fewshot) if args.fewshot else None
    prompts = [render.prompt_for(r["question"], system) for r in rows]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192,
        dtype="bfloat16",
        enforce_eager=False,
        generation_config="vllm",
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.n,
        seed=0,
        stop_token_ids=[1, 106],
    )
    outs = llm.generate(prompts, sp)

    n_correct = 0
    n_trunc = 0
    recs = []
    for r, o in zip(rows, outs):
        votes = {}
        first_text = None
        for c in o.outputs:
            if first_text is None:
                first_text = c.text
            if c.finish_reason == "length":
                n_trunc += 1
            _, ok = match_str(value=c.text, target=r["gold"], location="end", numeric=True)
            key = _
            votes[key] = votes.get(key, 0) + 1
        best = max(votes.items(), key=lambda kv: kv[1])[0]
        _, correct = match_str(value=best, target=r["gold"], location="exact", numeric=True)
        n_correct += bool(correct)
        recs.append(
            {
                "id": r["id"],
                "gold": r["gold"],
                "pred": best,
                "correct": bool(correct),
                "text": first_text,
            }
        )

    summary = {
        "model": args.model,
        "n": len(rows),
        "fewshot": args.fewshot,
        "temperature": args.temperature,
        "samples_per_item": args.n,
        "accuracy": n_correct / len(rows),
        "truncated_share": n_trunc / (len(rows) * args.n),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"summary": summary, "items": recs}, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
