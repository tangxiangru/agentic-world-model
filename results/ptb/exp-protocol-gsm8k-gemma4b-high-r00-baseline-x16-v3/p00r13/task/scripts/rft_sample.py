#!/usr/bin/env python3
"""Rejection-sampling data: sample k solutions per problem from a checkpoint,
keep the ones whose final answer matches gold, write them as SFT rows.

Prompts are built with build_sft_data.render_prompt, which verify_format.py
proves is byte-identical to what templates/gemma3.jinja produces for the
grader, so the samples are on-policy for the eval distribution.
Correctness is judged with inspect's own match_str(location="end",
numeric=True) - the same function that scores the benchmark.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft_data import render_prompt, sample_to_fewshot  # noqa: E402

from inspect_ai.scorer._common import match_str  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True, help="jsonl with {problem, answer}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    probs = [json.loads(l) for l in open(args.problems)]
    print(f"{len(probs)} problems x k={args.k}", flush=True)

    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    fewshot_pool = list(zip(gsm["question"], gsm["answer"]))

    prompts = [render_prompt(None, p["problem"]) for p in probs]

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=1536, dtype="bfloat16", seed=args.seed,
              enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    kept, n_correct, per_problem = [], 0, defaultdict(int)
    for p, o in zip(probs, outs):
        gold = str(p["answer"]).strip()
        texts = []
        for c in o.outputs:
            t = c.text.strip()
            if not t or t.count("ANSWER:") != 1:
                continue
            _, ok = match_str(t, gold, location="end", numeric=True)
            if not ok:
                continue
            n_correct += 1
            texts.append(t)
        # keep distinct correct samples at random: selecting by length would bias
        # the corpus toward terse chains and drop the harder problems' reasoning
        texts = sorted(set(texts))
        rng.shuffle(texts)
        texts = texts[: args.max_per_problem]
        for t in texts:
            if rng.random() < args.fewshot_frac:
                shots = rng.sample(fewshot_pool, 10)
                system = "\n\n".join(sample_to_fewshot(q, a) for q, a in shots)
            else:
                system = None
            kept.append({"prompt": render_prompt(system, p["problem"]),
                         "completion": t + "<end_of_turn>"})

    rng.shuffle(kept)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    solved = len({i for i, o in enumerate(outs)
                  if any(match_str(c.text.strip(), str(probs[i]["answer"]).strip(),
                                   location="end", numeric=True)[1] for c in o.outputs)})
    print(f"samples correct: {n_correct}/{len(probs)*args.k} "
          f"({n_correct/(len(probs)*args.k):.1%}); problems solved at least once: "
          f"{solved}/{len(probs)} ({solved/len(probs):.1%}); rows written: {len(kept)}")


if __name__ == "__main__":
    main()
