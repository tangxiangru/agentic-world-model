#!/usr/bin/env python3
"""Offline vLLM generation: local dev scoring and rejection-sampling.

The prompt is rendered by scripts/common.render_prompt, which is byte-for-byte
what templates/gemma3.jinja produces for the grader (verified in
analysis/template_match.txt). With --fewshot the same 10-shot system message
the harness builds is prepended, so a local dev number is comparable in kind to
the harness number - but it is measured on held-out GSM8K *train* items, never
on the benchmark test split.
"""
from __future__ import annotations

import argparse
import json
import os

from common import SNAPSHOT, graded_correct, render_prompt


def harness_fewshot_system(n: int = 10, seed: int = 42) -> str:
    """Reproduce inspect_evals.gsm8k's system message (train split, seed 42)."""
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

    shots = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=seed,
        limit=n,
    )
    return "\n\n".join(sample_to_fewshot(s) for s in shots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SNAPSHOT)
    ap.add_argument("--tokenizer", default=SNAPSHOT, help="Trainer checkpoints ship no tokenizer")
    ap.add_argument("--questions", required=True, help="jsonl with question (and gold)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        rows = rows[: args.limit]

    system = harness_fewshot_system(args.fewshot) if args.fewshot else None
    prompts = [render_prompt(r["question"], system) for r in rows]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=0 if args.temperature == 0 else None,
    )
    outs = llm.generate(prompts, sp)

    n_correct = n_total = 0
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            comps = [c.text for c in o.outputs]
            gold = str(r.get("gold", ""))
            oks = [graded_correct(c, gold) if gold else None for c in comps]
            if gold:
                n_total += 1
                n_correct += bool(oks[0])
            f.write(
                json.dumps(
                    {
                        "id": r.get("id"),
                        "question": r["question"],
                        "gold": gold,
                        "completions": comps,
                        "correct": oks,
                        "finish": [c.finish_reason for c in o.outputs],
                    }
                )
                + "\n"
            )
    if n_total:
        print(json.dumps({"greedy_pass@1": round(n_correct / n_total, 4), "n": n_total}))


if __name__ == "__main__":
    main()
