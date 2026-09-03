#!/usr/bin/env python3
"""Offline vLLM generation with the grader's exact prompt rendering.

Used for two things:
  * probe: greedy accuracy on data/dev_train500.jsonl (500 held-out gsm8k TRAIN
    problems) under the same 10-shot system message the grader uses -- a cheap
    n=500 read that never touches the benchmark test split;
  * sample: temperature sampling over training questions for rejection-sampling
    fine-tuning.

Prompts come from scripts/common.render_prompt, which was diffed byte-for-byte
against a jinja render of templates/gemma3.jinja.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import grader_reads, render_prompt  # noqa: E402


def load_fewshot_system() -> str:
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "fewshot10_system.txt")) as f:
        return f.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="jsonl with question + gold/answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fewshot", action="store_true", help="prepend the grader's 10-shot system message")
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            q = r.get("question") or r["problem"]
            a = str(r.get("gold") or r.get("answer") or r.get("expected_answer")).strip()
            rows.append({"id": r.get("id", len(rows)), "question": q, "gold": a})
            if args.limit and len(rows) >= args.limit:
                break

    system = load_fewshot_system() if args.fewshot else None
    prompts = [render_prompt(r["question"], system) for r in rows]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=4096,
        seed=args.seed,
        enforce_eager=False,
        dtype="bfloat16",
    )
    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],  # <eos>, <end_of_turn> -- the base generation_config list
        seed=args.seed if args.temperature == 0 else None,
    )
    outs = llm.generate(prompts, sp)

    n_corr = 0
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            gens = []
            for c in o.outputs:
                txt = c.text
                gens.append({"text": txt, "correct": grader_reads(txt, r["gold"]),
                             "ntok": len(c.token_ids), "finish": c.finish_reason})
            if gens[0]["correct"]:
                n_corr += 1
            f.write(json.dumps({"id": r["id"], "question": r["question"], "gold": r["gold"],
                                "gens": gens}) + "\n")
    print(json.dumps({
        "model": args.model,
        "n_items": len(rows),
        "first_sample_accuracy": n_corr / max(1, len(rows)),
        "temperature": args.temperature,
        "fewshot": args.fewshot,
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
