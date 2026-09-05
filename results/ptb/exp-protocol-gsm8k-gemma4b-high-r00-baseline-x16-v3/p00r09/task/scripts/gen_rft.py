#!/usr/bin/env python3
"""Sample k solutions per GSM8K-train question from a checkpoint, keep the ones
whose final ANSWER matches the gold answer, and write an SFT-shaped jsonl.

Questions come from the GSM8K *train* split and from OpenMathInstruct-2 rows
seeded on GSM8K train. The GSM8K test split is never read.
"""
import argparse
import json
import random
import re
from collections import defaultdict

import datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")
INT_RE = re.compile(r"^-?\d+$")


def final_answer(text):
    m = ANS_RE.search(text.strip())
    if not m:
        return None
    return m.group(1).replace(",", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-per-question", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n-aug", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    items = []
    for r in datasets.load_dataset("openai/gsm8k", "main", split="train"):
        a = r["answer"].split("####")[-1].strip().replace(",", "")
        if INT_RE.match(a):
            items.append((r["question"].strip(), a))
    n_gsm = len(items)

    if args.n_aug:
        ds = datasets.load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
        ds = ds.filter(lambda x: x["problem_source"] == "augmented_gsm8k", num_proc=8)
        seen = set()
        aug = []
        for r in ds:
            q = r["problem"].strip()
            a = r["expected_answer"].strip().replace(",", "")
            if q in seen or not INT_RE.match(a):
                continue
            seen.add(q)
            aug.append((q, a))
        rng.shuffle(aug)
        items += aug[:args.n_aug]
    print(f"questions: {n_gsm} gsm8k-train + {len(items)-n_gsm} augmented", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    template = open("/home/ben/task/templates/gemma3.jinja").read()
    prompts = []
    for q, _ in items:
        msgs = [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}]
        text = tok.apply_chat_template(msgs, chat_template=template, tokenize=False,
                                       add_generation_prompt=True)
        prompts.append({"prompt_token_ids": tok(text, add_special_tokens=False)["input_ids"]})

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, seed=args.seed, enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    kept = 0
    n_solved = 0
    with open(args.out, "w") as f:
        for (q, gold), o in zip(items, outs):
            good = []
            for c in o.outputs:
                txt = c.text.strip()
                a = final_answer(txt)
                if a is None:
                    continue
                try:
                    if abs(float(a) - float(gold)) > 1e-6:
                        continue
                except ValueError:
                    continue
                if txt.count("ANSWER:") != 1:
                    continue
                good.append(txt)
            if not good:
                continue
            n_solved += 1
            # prefer diverse, shorter solutions
            uniq = list(dict.fromkeys(good))
            uniq.sort(key=len)
            for txt in uniq[:args.max_per_question]:
                msgs = [{"role": "user",
                         "content": MATH_PROMPT_TEMPLATE.format(prompt=q)},
                        {"role": "assistant", "content": txt}]
                f.write(json.dumps({"messages": msgs,
                                    "completion": txt + "<end_of_turn>",
                                    "source": "rft_self", "answer": gold}) + "\n")
                kept += 1
    print(f"questions solved at least once: {n_solved}/{len(items)}; rows kept {kept}",
          flush=True)


if __name__ == "__main__":
    main()
