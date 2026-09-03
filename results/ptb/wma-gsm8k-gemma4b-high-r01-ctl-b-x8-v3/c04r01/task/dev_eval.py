#!/usr/bin/env python3
"""Fast local probe on held-out gsm8k TRAIN items (never the benchmark test split).

Reproduces the grader's prompt exactly: inspect_evals' 10-shot system message
(gsm8k train, seed 42) + MATH_PROMPT_TEMPLATE, rendered through the same
gemma3 chat wrapper, and inspect's match(numeric=True, location="end") scoring.
Also usable as a sampler for rejection-sampling fine-tuning (--k, --temperature).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

from build_data import MATH_PROMPT_TEMPLATE, render_prompt  # noqa: E402


def fewshot_system(n: int = 10, seed: int = 42) -> str:
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

    shots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                       sample_fields=record_to_sample, shuffle=True, seed=seed,
                       limit=n)
    return "\n\n".join(sample_to_fewshot(s) for s in shots)


NUM = re.compile(r"^-?[\d.]+$")


def last_number(text: str) -> str | None:
    v = text.strip().casefold().replace(",", "").replace("$", "")
    for w in reversed(re.split(r"\s+", v)):
        w = w.strip(".:;!?)*")
        if w.replace(".", "").isnumeric():
            try:
                return format(float(w), ".5g")
            except ValueError:
                return None
    return None


def norm(x: str) -> str:
    try:
        return format(float(str(x).replace(",", "")), ".5g")
    except ValueError:
        return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/dev_train250.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--fewshot", type=int, default=10)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    sys_prefix = (fewshot_system(args.fewshot) + "\n\n") if args.fewshot else ""
    prompts = [render_prompt(r["question"], sys_prefix) for r in rows]

    # vLLM's worker bootstrap re-parses sys.argv; hide our flags from it
    sys.argv = sys.argv[:1]
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              max_model_len=4096, enforce_eager=False, dtype="bfloat16")
    sp = SamplingParams(n=args.k, temperature=args.temperature,
                        top_p=1.0 if args.temperature == 0 else 0.95,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    n_ok = 0
    recs = []
    for r, o in zip(rows, outs):
        gold = norm(r["gold"])
        texts = [c.text for c in o.outputs]
        preds = [last_number(t) for t in texts]
        hit = any(p == gold for p in preds)
        n_ok += bool(hit)
        recs.append({"id": r.get("id"), "question": r["question"], "gold": r["gold"],
                     "preds": preds, "texts": texts,
                     "finish": [c.finish_reason for c in o.outputs]})
    acc = n_ok / max(1, len(rows))
    print(f"model={args.model} n={len(rows)} k={args.k} T={args.temperature} "
          f"acc(any-of-k)={acc:.4f}")
    trunc = sum(1 for r in recs for f in r["finish"] if f == "length") / max(
        1, sum(len(r["finish"]) for r in recs))
    print(f"truncated share={trunc:.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "n": len(rows), "k": args.k,
                       "temperature": args.temperature, "accuracy": acc,
                       "truncated": trunc, "records": recs}, f)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
