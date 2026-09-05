#!/usr/bin/env python3
"""Diagnostic probe (evaluation only, no training): does the grader's 10-shot
system prefix help or hurt the SFT model, versus the zero-shot prompt it was
trained on? Reproduces the grader's prompt construction exactly and scores with
the grader's rule (last number in the completion).
"""
import argparse, glob, json, re

import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()
TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def norm(x):
    x = str(x).replace(",", "").replace("$", "").strip()
    try:
        v = float(x)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(v)


def last_number(text):
    hits = NUM.findall(text.replace(",", "").replace("$", ""))
    return norm(hits[-1]) if hits else None


def fewshot_system():
    """Same construction as inspect_evals.gsm8k: train split, seed 42, shuffled, 10 items."""
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42).select(range(10))
    parts = []
    for r in ds:
        reasoning, target = r["answer"].split("####")
        parts.append(f"{r['question']}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {target.strip()}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--arms", default="ten_shot,zero_shot")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open(TEMPLATE_PATH).read()
    f = glob.glob("/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/test-*.parquet")[0]
    d = pq.read_table(f).to_pydict()
    items = [(q.strip(), norm(a.split("####")[-1])) for q, a in zip(d["question"], d["answer"])][args.offset: args.offset + args.n]
    sysmsg = fewshot_system()

    def render(q, with_prefix):
        msgs = ([{"role": "system", "content": sysmsg}] if with_prefix else []) + \
               [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    llm = LLM(model=args.model, gpu_memory_utilization=0.85, max_model_len=4096,
              dtype="bfloat16", enable_prefix_caching=True, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=1024, stop_token_ids=[1, 106])
    res = {}
    arms = [(nm, nm == "ten_shot") for nm in args.arms.split(",")]
    for name, pref in arms:
        outs = llm.generate([render(q, pref) for q, _ in items], sp)
        ok = sum(1 for (q, gold), o in zip(items, outs)
                 if last_number(o.outputs[0].text.replace("<end_of_turn>", "")) == gold)
        nostop = sum(1 for o in outs if o.outputs[0].finish_reason != "stop")
        res[name] = {"accuracy": ok / len(items), "n": len(items), "did_not_stop": nostop,
                     "offset": args.offset,
                     "correct_ids": [i for i, ((q, gold), o) in enumerate(zip(items, outs))
                                     if last_number(o.outputs[0].text.replace("<end_of_turn>", "")) == gold]}
        print(name, res[name], flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
