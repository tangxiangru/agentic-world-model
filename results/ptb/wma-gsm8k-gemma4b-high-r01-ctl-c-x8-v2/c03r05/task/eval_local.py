#!/usr/bin/env python3
"""Score a checkpoint on the local GSM8K-*train* holdout probe.

This is a diagnostic, not the protocol: the protocol is evaluate.py --limit 150
on the official test split. This script exists to answer questions the protocol
cannot answer cheaply (does the 10-shot prefix help or hurt? which items does a
checkpoint get wrong?) without touching the benchmark test set.

Prompts are built with the grader's own templates/gemma3.jinja and
inspect_evals' MATH_PROMPT_TEMPLATE, and graded the way
inspect_ai match(numeric=True, location="end") grades: last numeric
whitespace-token of the completion.
"""
from __future__ import annotations

import argparse
import json
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def norm_num(tok: str) -> str | None:
    t = tok.strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", t):
        return format(float(t), ".5g")
    return None


def graded_answer(completion: str) -> str | None:
    for w in reversed(completion.strip().split()):
        n = norm_num(w)
        if n is not None:
            return n
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/dev_train_holdout.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--fewshot", type=int, default=1,
                    help="1 = prepend the grader's exact 10-shot system message")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open("templates/gemma3.jinja").read()
    fewshot = open("data/fewshot_system.txt").read()

    items = [json.loads(l) for l in open(args.data)][: args.n]
    prompts = []
    for it in items:
        msgs = []
        if args.fewshot:
            msgs.append({"role": "system", "content": fewshot})
        msgs.append({"role": "user",
                     "content": MATH_PROMPT_TEMPLATE.format(prompt=it["question"])})
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              max_model_len=4096, dtype="bfloat16", enforce_eager=False)
    sp = SamplingParams(temperature=args.temp, top_p=1.0 if args.temp == 0 else 0.95,
                        max_tokens=args.max_tokens, n=args.k,
                        stop_token_ids=[106, 1])
    outs = llm.generate(prompts, sp)

    n_ok = n_trunc = 0
    recs = []
    for it, o in zip(items, outs):
        cands = [c.text for c in o.outputs]
        got = [graded_answer(c) for c in cands]
        gold = norm_num(it["gold"])
        ok = any(g == gold for g in got)
        n_ok += ok
        n_trunc += sum(1 for c in o.outputs if c.finish_reason == "length")
        recs.append({"id": it["id"], "gold": it["gold"], "got": got,
                     "correct": bool(ok), "completion": cands[0]})
    res = {"model": args.model, "n": len(items), "fewshot": args.fewshot,
           "k": args.k, "temp": args.temp,
           "accuracy": n_ok / len(items), "unterminated": n_trunc}
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": res, "samples": recs}, f)


if __name__ == "__main__":
    main()
