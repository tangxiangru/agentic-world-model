#!/usr/bin/env python3
"""Rejection sampling: draw k solutions per training question from a checkpoint,
keep the ones whose final ANSWER matches the reference answer.

Questions come from GSM8K *train* and from OpenMathInstruct-2's gsm8k-family
problems (both carry a reference answer); the test split is never touched.
Prompts are rendered with the same code the grader's template produces.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_fmt import fmt_number, render_prompt  # noqa: E402

ANS = re.compile(r"ANSWER:\s*(-?[\d,]+(?:\.\d+)?)")


def last_answer(text: str):
    m = ANS.findall(text)
    if not m:
        return None
    return fmt_number(m[-1])


def load_questions(n_omi: int, seed: int):
    from datasets import load_dataset
    import re as _re

    out = []
    for name in ("main",):
        ds = load_dataset("openai/gsm8k", name, split="train")
        for r in ds:
            a = r["answer"].rsplit("####", 1)[-1]
            v = fmt_number(a)
            if v is not None:
                out.append({"q": r["question"], "a": v, "src": "gsm8k-train"})
    if n_omi > 0:
        ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
        seen = set()
        pool = []
        for r in ds:
            if r["problem_source"] != "augmented_gsm8k":
                continue
            p = r["problem"]
            if p in seen:
                continue
            seen.add(p)
            v = fmt_number(r["expected_answer"])
            if v is None:
                continue
            pool.append({"q": p, "a": v, "src": "augmented_gsm8k"})
        random.Random(seed).shuffle(pool)
        out.extend(pool[:n_omi])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n-omi", type=int, default=20000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    qs = load_questions(args.n_omi, args.seed)
    if args.limit:
        qs = qs[: args.limit]
    print(f"questions: {len(qs)}", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=1536,
        enable_prefix_caching=True,
        seed=args.seed,
        generation_config="vllm",
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=None,
    )
    prompts = [render_prompt(x["q"]) for x in qs]
    outs = llm.generate(prompts, sp)

    n_kept = 0
    n_any = 0
    with open(args.out, "w") as f:
        for x, o in zip(qs, outs):
            texts = []
            for c in o.outputs:
                t = c.text.strip()
                if last_answer(t) == x["a"] and t.count("ANSWER:") == 1:
                    texts.append(t)
            if texts:
                n_any += 1
            n_kept += len(texts)
            f.write(json.dumps({"q": x["q"], "a": x["a"], "src": x["src"],
                                "solutions": texts,
                                "n_sampled": len(o.outputs)}) + "\n")
    print(f"questions with >=1 correct sample: {n_any}/{len(qs)}; "
          f"total correct samples: {n_kept}", flush=True)


if __name__ == "__main__":
    main()
