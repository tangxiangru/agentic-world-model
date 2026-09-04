#!/usr/bin/env python3
"""Batched vLLM generation in the grader's exact rendering.

Used for two things:
  * fast diagnostics on questions that are NOT the benchmark test set
    (evaluate.py at --max-connections 2 takes ~15 min per 150 items);
  * rejection sampling for RFT.

The prompt is built by common.render_prompt, i.e. templates/gemma3.jinja with
the grader's MATH_PROMPT_TEMPLATE, so what the model sees here is what it sees
at grading time.
"""
from __future__ import annotations

import argparse
import json
import re

from transformers import AutoTokenizer

import common

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")


def extract(text: str) -> str | None:
    m = ANS_RE.findall(text)
    if not m:
        return None
    a = m[-1].replace(",", "")
    if "." in a:
        a = a.rstrip("0").rstrip(".")
    return a or None


def same(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a == b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with question,answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--fewshot", action="store_true",
                    help="render with the grader's 10-shot system message")
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit > 0:
        rows = rows[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [common.render_prompt(tok, r["question"], args.fewshot) for r in rows]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              max_model_len=4096, dtype="bfloat16", seed=args.seed,
              generation_config="vllm", enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    n_correct_any, n_correct_first, n_stop = 0, 0, 0
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            gold = str(r["answer"]).strip()
            cands = []
            for c in o.outputs:
                got = extract(c.text)
                ok = same(got, gold)
                n_stop += c.finish_reason == "stop"
                cands.append({"text": c.text, "pred": got, "correct": ok})
            n_correct_any += any(c["correct"] for c in cands)
            n_correct_first += cands[0]["correct"]
            f.write(json.dumps({"question": r["question"], "answer": gold,
                                "candidates": cands}) + "\n")
    n = len(rows)
    print(json.dumps({
        "n": n, "k": args.k, "temperature": args.temperature,
        "acc_first_sample": n_correct_first / n,
        "pass_at_k": n_correct_any / n,
        "stop_share": n_stop / (n * args.k),
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
