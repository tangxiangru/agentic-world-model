#!/usr/bin/env python3
"""Fast offline probe on the held-out GSM8K-TRAIN slice (never trained on).

Reproduces the harness prompt exactly: the 10-shot system message built by
inspect_evals.gsm8k (seed 42) folded into the first user turn by
templates/gemma3.jinja, plus MATH_PROMPT_TEMPLATE, and the same
"last number in the completion" grading rule as inspect_ai's
match(location='end', numeric=True).
"""
from __future__ import annotations

import argparse
import json
import re

from transformers import AutoTokenizer

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def graded_correct(completion: str, gold: str) -> bool:
    """The grader itself: inspect_ai's match(location='end', numeric=True)."""
    from inspect_ai.scorer._common import match_str

    _, ok = match_str(value=completion, target=gold, location="end", numeric=True)
    return bool(ok)


def last_number(text: str) -> str | None:
    """The number the grader would read: last numeric token, normalised."""
    v = re.sub(r"[,\$]", "", text.strip().casefold())
    words = re.split(r"\s+", v)
    words.reverse()
    for w in words:
        w2 = w.strip(".:;!?)(\"'*")
        if w2.replace(".", "").replace("-", "").isnumeric():
            try:
                return format(float(w2), ".5g")
            except ValueError:
                continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="/home/ben/task/data/probe_train300.jsonl")
    ap.add_argument("--fewshot", default="/home/ben/task/data/fewshot_system.txt")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-fewshot", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)][: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open("/home/ben/task/templates/gemma3.jinja").read()
    system = None if args.no_fewshot else open(args.fewshot).read()

    prompts = []
    for r in rows:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=r["question"])})
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
        dtype="bfloat16",
        enforce_eager=False,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
    )
    outs = llm.generate(prompts, sp)

    n_ok = 0
    recs = []
    n_trunc = 0
    for r, o in zip(rows, outs):
        text = o.outputs[0].text
        pred = last_number(text)
        gold = r["gold"].replace(",", "")
        ok = graded_correct(text, gold)
        n_ok += ok
        if o.outputs[0].finish_reason == "length":
            n_trunc += 1
        recs.append(
            {
                "id": r["id"],
                "gold": gold,
                "pred": pred,
                "correct": bool(ok),
                "n_tokens": len(o.outputs[0].token_ids),
                "finish": o.outputs[0].finish_reason,
                "completion": text,
            }
        )
    acc = n_ok / len(rows)
    print(f"probe accuracy {acc:.4f}  ({n_ok}/{len(rows)})  truncated={n_trunc}")
    with open(args.out, "w") as f:
        json.dump({"accuracy": acc, "n": len(rows), "truncated": n_trunc, "records": recs}, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
