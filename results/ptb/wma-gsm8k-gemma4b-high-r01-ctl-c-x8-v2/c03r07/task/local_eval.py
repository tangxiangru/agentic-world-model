#!/usr/bin/env python3
"""Fast offline scorer on the held-out dev300 (gsm8k TRAIN items, never trained on).

Mirrors the graded protocol exactly where it matters: the same 10-shot system
prefix, the same MATH_PROMPT_TEMPLATE, the same gemma3.jinja rendering (via
train_sft.render), and the same scoring rule as inspect's
match(numeric=True, location='end') - the LAST number in the completion.

This is a diagnostic, not a comparator: the headline metric stays
`python evaluate.py --limit 150`.
"""
from __future__ import annotations

import argparse
import json
import re

from train_sft import render

FEWSHOT = open("data/fewshot_system_message.txt").read()
TEMPLATE = open("data/math_prompt_template.txt").read()
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str) -> str | None:
    nums = NUM.findall(text.replace("$", ""))
    if not nums:
        return None
    v = nums[-1].replace(",", "").rstrip(".")
    return v


def normalize(x: str) -> str:
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return x


def build_prompt(question: str, fewshot: bool) -> str:
    body = TEMPLATE.format(prompt=question).strip()
    return f"{FEWSHOT}\n\n{body}" if fewshot else body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/dev300.jsonl")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--no-fewshot", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.data)][: args.limit]
    prompts = [render(build_prompt(r["question"], not args.no_fewshot), None) for r in rows]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        generation_config="vllm",
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    outs = llm.generate(prompts, sp)

    n_correct = 0
    n_stopped = 0
    recs = []
    for r, o in zip(rows, outs):
        text = o.outputs[0].text
        stopped = o.outputs[0].finish_reason == "stop"
        pred = last_number(text)
        ok = pred is not None and normalize(pred) == normalize(r["gold"])
        n_correct += ok
        n_stopped += stopped
        recs.append({"id": r["id"], "gold": r["gold"], "pred": pred, "correct": bool(ok),
                     "stopped": stopped, "n_tokens": len(o.outputs[0].token_ids)})

    summary = {
        "model": args.model,
        "n": len(rows),
        "accuracy": n_correct / len(rows),
        "stopped_share": n_stopped / len(rows),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "fewshot": not args.no_fewshot,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "items": recs}, f, indent=2)


if __name__ == "__main__":
    main()
