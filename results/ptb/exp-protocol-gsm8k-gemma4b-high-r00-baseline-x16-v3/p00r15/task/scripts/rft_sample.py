#!/usr/bin/env python3
"""Rejection-sampling data generation: sample k solutions per training problem from
our own SFT checkpoint, keep the ones whose final answer is right.

Problems come from the GSM8K *train* split and from OpenMathInstruct-2's
gsm8k/augmented_gsm8k problems (also train-derived). Gold answers come with them;
nothing here touches the benchmark test set.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys

sys.path.insert(0, "/home/ben/task/scripts")
from format_utils import STOP_TOKEN, random_fewshot_system, render_prompt  # noqa: E402

NUMLIKE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def last_number(text: str) -> str | None:
    for w in reversed(re.split(r"\s+", text.strip())):
        c = w.strip().replace("$", "").replace(",", "").rstrip(".").rstrip("%")
        if c and NUMLIKE.match(c):
            c = c.rstrip(".")
            if c.endswith(".0"):
                c = c[:-2]
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", default="/home/ben/task/data/pool_big.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-problems", type=int, default=40000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)

    # one entry per distinct problem
    seen: dict[str, dict] = {}
    for line in open(args.pool):
        r = json.loads(line)
        if r["problem"] not in seen:
            seen[r["problem"]] = r
    probs = list(seen.values())
    rng.shuffle(probs)
    probs = probs[: args.n_problems]
    print(f"[rft] {len(probs)} distinct problems, k={args.k}", flush=True)

    prompts = [render_prompt(tok, r["prompt"]) for r in probs]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=2048,
              dtype="bfloat16", enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    kept, n_gen, n_ok = [], 0, 0
    per_problem_hits = []
    for r, o in zip(probs, outs):
        gold = r["answer"]
        hits = 0
        texts = set()
        for c in o.outputs:
            n_gen += 1
            t = c.text.strip()
            if not t or "ANSWER:" not in t:
                continue
            if last_number(t) != gold:
                continue
            if t.count("ANSWER:") != 1:
                continue
            hits += 1
            n_ok += 1
            if t in texts:
                continue
            texts.add(t)
            if len(texts) <= 2:  # at most 2 distinct correct solutions per problem
                kept.append({"problem": r["problem"], "prompt": r["prompt"],
                             "completion": t, "answer": gold, "source": "rft:self"})
        per_problem_hits.append(hits)

    solved = sum(1 for h in per_problem_hits if h > 0)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"[rft] generated {n_gen}, correct {n_ok} ({n_ok/max(1,n_gen):.3f}), "
          f"problems solved at least once {solved}/{len(probs)} ({solved/len(probs):.3f}), "
          f"kept {len(kept)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
