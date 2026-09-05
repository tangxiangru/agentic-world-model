#!/usr/bin/env python3
"""Score a checkpoint on the held-out probe set (GSM8K *train* problems that
build_data.py excludes from the corpus).

Uses the grader's own templates/gemma3.jinja and the grader's scoring rule
(match numeric at the end: the last whitespace-separated number in the
completion), so a probe number moves for the same reasons the harness number
moves. Decoding is greedy, matching the generation_config.json written into the
trained checkpoints.
"""
import argparse
import json
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

TASK = "/home/ben/task"

MATH_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str):
    words = re.split(r"\s+", text.strip())
    for w in reversed(words):
        w = w.replace(",", "").rstrip(".").lstrip("$")
        m = NUM.fullmatch(w)
        if m:
            v = w
            if v.endswith(".0"):
                v = v[:-2]
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", default=f"{TASK}/data/probe250.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--fewshot", type=int, default=0,
                    help="prepend the grader's own k-shot block (same source, seed and "
                         "formatting inspect_evals.gsm8k uses) to every prompt")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.probe)]
    if args.n:
        items = items[: args.n]

    prefix = ""
    if args.fewshot:
        # byte-identical to the system message inspect_evals.gsm8k builds
        from inspect_ai.dataset import hf_dataset
        from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot
        shots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                           sample_fields=record_to_sample, shuffle=True, seed=42,
                           limit=args.fewshot)
        prefix = "\n\n".join(sample_to_fewshot(s) for s in shots) + "\n\n"

    tok = AutoTokenizer.from_pretrained(args.model)
    tpl = open(f"{TASK}/templates/gemma3.jinja").read()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user",
              "content": prefix + MATH_PROMPT_TEMPLATE.format(prompt=it["question"])}],
            chat_template=tpl, tokenize=False, add_generation_prompt=True)
        for it in items
    ]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, enforce_eager=False, dtype="bfloat16")
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    recs, n_ok = [], 0
    for it, o in zip(items, outs):
        text = o.outputs[0].text
        pred = last_number(text)
        ok = pred is not None and pred == it["gold"]
        n_ok += ok
        recs.append({"id": it["id"], "gold": it["gold"], "pred": pred, "correct": bool(ok),
                     "n_out_tokens": len(o.outputs[0].token_ids), "output": text})

    acc = n_ok / len(items)
    json.dump({"model": args.model, "n": len(items), "fewshot": args.fewshot, "accuracy": acc,
               "mean_out_tokens": sum(r["n_out_tokens"] for r in recs) / len(recs),
               "samples": recs}, open(args.out, "w"), indent=1)
    print(f"probe accuracy {acc:.4f} on n={len(items)} -> {args.out}")


if __name__ == "__main__":
    main()
