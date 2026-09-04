#!/usr/bin/env python3
"""Rejection-sampling fine-tuning data: sample k solutions per training problem
from the current checkpoint, keep the ones whose final number matches gold.

Problems come from the SFT pool's own questions (GSM8K train + OpenMathInstruct-2
augmented-GSM8K variants) -- never from the benchmark test copy.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402
from eval_local import last_number, norm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True, help="jsonl with question/answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--hard-only", action="store_true",
                    help="keep only problems where at least one sample is wrong")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    items = [json.loads(l) for l in open(args.problems)]
    if args.limit:
        items = items[: args.limit]
    print(f"[rft] {len(items)} problems x k={args.k}", flush=True)

    prompts = [render.build_prompt(it["question"]) for it in items]

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, dtype="bfloat16")
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    import datasets
    tr = datasets.load_dataset("openai/gsm8k", "main", split="train")

    def fewshots(k):
        idxs = rng.sample(range(len(tr)), k)
        bl = []
        for i in idxs:
            r = tr[i]
            parts = r["answer"].split("####")
            bl.append(render.fewshot_block(r["question"], "####".join(parts[:-1]).strip(),
                                           parts[-1].strip()))
        return bl

    n_written = 0
    stats = defaultdict(int)
    with open(args.out, "w") as fh:
        for it, o in zip(items, outs):
            gold = it["answer"]
            good, n_ok = [], 0
            for c in o.outputs:
                txt = c.text.strip()
                pred = last_number(txt)
                ok = pred is not None and norm(pred) == norm(gold)
                n_ok += ok
                if not ok or c.finish_reason == "length":
                    continue
                if txt.count("ANSWER:") != 1 or not txt.endswith(gold):
                    continue
                good.append(txt)
            stats[f"nok_{n_ok}"] += 1
            if not good:
                continue
            if args.hard_only and n_ok == args.k:
                continue
            seen = set()
            uniq = []
            for g in good:
                key = " ".join(g.split())
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(g)
            uniq.sort(key=len)  # prefer the shortest correct chains
            for g in uniq[: args.max_per_problem]:
                k = rng.choice([1, 2, 3, 4, 10]) if rng.random() < args.fewshot_frac else 0
                fh.write(json.dumps({
                    "prompt": render.build_prompt(it["question"], fewshots(k) if k else None),
                    "completion": render.build_completion(g),
                    "answer": gold, "source": "rft:self", "nshot": k}) + "\n")
                n_written += 1
    print(f"[rft] wrote {n_written} rows -> {args.out}")
    print("[rft] correct-of-k histogram:", dict(sorted(stats.items())), flush=True)


if __name__ == "__main__":
    main()
