#!/usr/bin/env python3
"""Rejection sampling (STaR): generate multiple solutions per GSM8K train question
with a fine-tuned model via vLLM, keep only those whose final answer matches gold."""
import argparse
import json
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def last_number(text):
    # mimic inspect match(numeric=True, location='end'): last number in output
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")


def norm(x):
    x = str(x).replace(",", "").strip()
    try:
        f = float(x)
        if f == int(f):
            return str(int(f))
        return str(f)
    except Exception:
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="star_data.jsonl")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max_tokens", type=int, default=640)
    ap.add_argument("--keep_per_q", type=int, default=2)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()

    with open("templates/gemma3.jinja") as f:
        chat_template = f.read()
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = chat_template

    ds = load_dataset("openai/gsm8k", "main", split="train")
    prompts, metas = [], []
    for rec in ds:
        q = rec["question"].strip()
        gold = norm(rec["answer"].split("####")[-1].strip())
        user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        text = tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
        metas.append((q, gold, user))

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=1536, enable_prefix_caching=True)
    sp = SamplingParams(n=args.n, temperature=args.temp, top_p=0.95,
                        max_tokens=args.max_tokens, stop=["<end_of_turn>"])

    outputs = llm.generate(prompts, sp)

    kept = 0
    n_q_solved = 0
    with open(args.out, "w") as fout:
        for (q, gold, user), out in zip(metas, outputs):
            good = []
            seen = set()
            for comp in out.outputs:
                txt = comp.text.strip()
                pred = last_number(txt)
                if pred is None:
                    continue
                if norm(pred) == gold:
                    # normalize: ensure ends with ANSWER: gold
                    key = txt[:120]
                    if key in seen:
                        continue
                    seen.add(key)
                    # ensure clean ending
                    if "ANSWER:" not in txt:
                        txt = txt + f"\n\nANSWER: {gold}"
                    good.append(txt)
            if good:
                n_q_solved += 1
            for txt in good[: args.keep_per_q]:
                fout.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": txt},
                    ],
                    "question": q, "target": gold,
                }) + "\n")
                kept += 1
    print(f"solved {n_q_solved}/{len(metas)} questions; kept {kept} solutions -> {args.out}")


if __name__ == "__main__":
    main()
