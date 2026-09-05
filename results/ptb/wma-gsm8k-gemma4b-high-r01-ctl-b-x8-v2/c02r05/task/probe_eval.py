"""Score a checkpoint on the held-out GSM8K-train probe set.

Uses the grader's prompt (10-shot system message + MATH_PROMPT_TEMPLATE through
templates/gemma3.jinja) and the grader's scoring rule (last numeric token of the
completion, match(numeric=True, location="end")). It is a stand-in for the
official eval on items that are not benchmark test items, so it can be used as
a watch set without touching the test copy.
"""
from __future__ import annotations

import argparse
import json
import re

from prompt_fmt import FEWSHOT_SYSTEM, render_prompt


def last_number(text: str) -> str | None:
    v = text.strip().replace(",", "").replace("$", "")
    words = re.split(r"\s+", v)
    for w in reversed(words):
        w = w.strip(".:!?)")
        if w.replace(".", "").replace("-", "").isnumeric():
            return w
    return None


def norm(x: str | None) -> str | None:
    if x is None:
        return None
    try:
        return format(float(x), ".5g")
    except ValueError:
        return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", default="analysis/probe250.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.probe)]
    if args.limit:
        rows = rows[: args.limit]

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, dtype="bfloat16", seed=args.seed)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_tokens,
                        stop_token_ids=[1, 106], seed=args.seed)
    prompts = [render_prompt(r["question"], FEWSHOT_SYSTEM) for r in rows]
    outs = llm.generate(prompts, sp)

    results, n_ok, n_trunc = [], 0, 0
    for r, o in zip(rows, outs):
        c = o.outputs[0]
        text = c.text
        ok = norm(last_number(text)) == norm(r["gold"])
        n_ok += ok
        n_trunc += c.finish_reason == "length"
        results.append({"id": r["id"], "gold": r["gold"], "correct": bool(ok),
                        "finish_reason": c.finish_reason, "output": text})
    summary = {"model": args.model, "n": len(rows), "accuracy": n_ok / len(rows),
               "truncated": n_trunc, "temperature": args.temperature}
    print(json.dumps(summary, indent=2), flush=True)
    json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
