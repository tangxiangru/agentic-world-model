#!/usr/bin/env python3
"""Cheap offline probe on GSM8K TRAIN items (never the test split).

Reproduces the harness's decoding contract - gemma3.jinja, MATH_PROMPT_TEMPLATE,
stop on <end_of_turn>, score = last number in the completion - so a checkpoint can be
compared 0-shot vs 10-shot without spending a harness eval.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

MATH_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def norm(v: str) -> str:
    v = v.replace(",", "")
    if "." in v:
        v = v.rstrip("0").rstrip(".")
    return v or "0"


def last_number(t: str) -> str | None:
    ms = NUM.findall(t)
    return norm(ms[-1]) if ms else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--shots", type=int, nargs="+", default=[0, 10])
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ds = load_dataset("openai/gsm8k", "main", split="train")
    n_all = len(ds)
    probe = [ds[i] for i in range(n_all - args.n, n_all)]
    rng = random.Random(42)
    shot_src = [ds[i] for i in rng.sample(range(0, n_all - args.n), 10)]
    shots_text = []
    for r in shot_src:
        parts = r["answer"].split("####")
        shots_text.append(
            f"{r['question']}\n\nReasoning:\n" + "####".join(parts[:-1]).strip() + f"\n\nANSWER: {parts[-1].strip()}"
        )

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open("/home/ben/task/templates/gemma3.jinja").read()

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac, max_model_len=4096, dtype="bfloat16")
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop_token_ids=[1, 106])

    results = {}
    for k in args.shots:
        prompts = []
        for r in probe:
            msgs = []
            if k:
                msgs.append({"role": "system", "content": "\n\n".join(shots_text[:k])})
            msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=r["question"])})
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        outs = llm.generate(prompts, sp)
        correct = 0
        fmt_ok = 0
        for r, o in zip(probe, outs):
            t = o.outputs[0].text.strip()
            gold = norm(r["answer"].split("####")[-1].strip())
            if re.search(r"ANSWER:\s*\$?-?[\d,]+(\.\d+)?\.?$", t):
                fmt_ok += 1
            if last_number(t) == gold:
                correct += 1
        results[f"{k}shot"] = {
            "accuracy": correct / len(probe),
            "format_ok": fmt_ok / len(probe),
            "n": len(probe),
        }
        print(k, "shot ->", results[f"{k}shot"], flush=True)

    json.dump(results, open(args.out, "w"), indent=2)
    print("PROBE", json.dumps(results))


if __name__ == "__main__":
    main()
