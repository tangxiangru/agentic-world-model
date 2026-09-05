#!/usr/bin/env python3
"""Score a checkpoint on the GSM8K-train holdout watch set (greedy, zero-shot).

Not a substitute for the official protocol: this is the internal probe defined
in the cards' problem.watch_set, on items that never entered training.
"""
import argparse
import json
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

TEMPLATE = "/home/ben/task/templates/gemma3.jinja"
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()
ANS = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--items", default="data/dev_train_holdout.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
args = ap.parse_args()

items = [json.loads(l) for l in open(args.items)]
tok = AutoTokenizer.from_pretrained(args.model)
template = open(TEMPLATE).read()
prompts = [tok.apply_chat_template(
    [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=it["question"])}],
    chat_template=template, tokenize=False, add_generation_prompt=True) for it in items]

llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
          max_model_len=2048, enable_prefix_caching=True)
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=768,
                                            stop_token_ids=[1, 106]))
res = []
for it, o in zip(items, outs):
    text = o.outputs[0].text
    m = ANS.search(text)
    pred = m.group(1).replace(",", "").rstrip(".") if m else None
    res.append({"id": it["id"], "gold": it["gold"], "pred": pred,
                "correct": pred is not None and pred.split(".")[0] == it["gold"],
                "output": text})
acc = sum(r["correct"] for r in res) / len(res)
json.dump({"model": args.model, "n": len(res), "accuracy": acc, "items": res},
          open(args.out, "w"), indent=1)
print(json.dumps({"model": args.model, "n": len(res), "accuracy": acc}, indent=1))
