#!/usr/bin/env python3
"""Sample k solutions per question from a checkpoint with vLLM, for
rejection-sampling fine-tuning (RFT) or for scoring a private dev set.

Prompts are rendered with the same renderer the trainer and the grader use
(scripts/eval_format.py -> templates/gemma3.jinja), including the grader's exact
10-shot system prefix when --fewshot is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, "/home/ben/task/scripts")
from eval_format import gsm8k_fewshot_system, render, user_prompt  # noqa: E402

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text: str) -> str | None:
    m = NUM.findall(text.replace(",", ""))
    if not m:
        return None
    v = m[-1]
    if v.endswith(".0"):
        v = v[:-2]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with id/question/gold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fewshot", action="store_true")
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        rows = rows[: args.limit]
    sysmsg = gsm8k_fewshot_system() if args.fewshot else None
    prompts = [render(sysmsg, user_prompt(r["question"])) for r in rows]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=4096,
        dtype="bfloat16",
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=0,
    )
    outs = llm.generate(prompts, sp)

    n_corr = 0
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            samples = []
            for c in o.outputs:
                txt = c.text
                pred = last_number(txt)
                ok = pred is not None and pred == str(r["gold"]).replace(",", "")
                samples.append({"text": txt, "pred": pred, "correct": ok,
                                "finished": c.finish_reason == "stop"})
            n_corr += any(s["correct"] for s in samples)
            f.write(json.dumps({"id": r["id"], "question": r["question"],
                                "gold": r["gold"], "samples": samples}) + "\n")
    print(f"wrote {args.out}; pass@{args.k} = {n_corr}/{len(rows)} = {n_corr/len(rows):.4f}")


if __name__ == "__main__":
    main()
