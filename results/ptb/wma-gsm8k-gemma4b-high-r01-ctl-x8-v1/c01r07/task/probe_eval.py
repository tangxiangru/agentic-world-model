#!/usr/bin/env python3
"""Cheap local dev signal: greedy accuracy on data/probe300.jsonl.

The 300 questions come from the gsm8k TRAIN split and are excluded from every
training file, so this is a legitimate held-out probe (protocol rule 7). It uses
the grader's prompt and the grader's answer-reading rule, but it is NOT the
official protocol - card measurements must still come from evaluate.py.
"""
from __future__ import annotations

import argparse
import json

import prompt_spec as ps
from rft_sample import last_number, norm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", default="data/probe300.jsonl")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--fewshot", action="store_true", default=True)
    ap.add_argument("--no-fewshot", dest="fewshot", action="store_false")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.probe)][: args.limit]
    sysmsg = ps.fewshot_system_message() if args.fewshot else None
    prompts = [ps.render_prompt(r["question"], sysmsg) for r in rows]

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, enable_prefix_caching=True)
    outs = llm.generate(prompts, SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens, stop_token_ids=[1, 106]))

    correct = capped = no_marker = 0
    fails = []
    for r, o in zip(rows, outs):
        txt = o.outputs[0].text
        if o.outputs[0].finish_reason == "length":
            capped += 1
        if "ANSWER:" not in txt:
            no_marker += 1
        ok = last_number(txt) == norm(r["gold"])
        if ok:
            correct += 1
        elif len(fails) < 40:
            fails.append({"id": r["id"], "gold": r["gold"], "question": r["question"],
                          "output": txt[-800:]})

    summary = {"model": args.model, "n": len(rows), "fewshot": args.fewshot,
               "accuracy": correct / len(rows), "hit_cap": capped,
               "no_answer_marker": no_marker}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "failures": fails}, f, indent=2)


if __name__ == "__main__":
    main()
