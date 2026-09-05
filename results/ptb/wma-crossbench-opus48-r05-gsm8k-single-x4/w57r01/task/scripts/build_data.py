#!/usr/bin/env python3
"""Build SFT data from the GSM8K TRAIN split (disjoint from the test set).

Each row: prompt (rendered with the SAME gemma3.jinja the grader uses,
add_generation_prompt=True, zero-shot) + completion (clean CoT ending in
'ANSWER: N' then <end_of_turn>). Completion-only loss is applied in training.
"""
import argparse, json, os, re

MATH_PROMPT_TEMPLATE = (
    'Solve the following math problem step by step. The last line of your '
    'response should be of the form "ANSWER: $ANSWER" (without quotes) where '
    '$ANSWER is the answer to the problem.\n\n{prompt}\n\nRemember to put your '
    'answer on its own line at the end in the form "ANSWER: $ANSWER" (without '
    'quotes) where $ANSWER is the answer to the problem, and you do not need to '
    'use a \\boxed command.\n\nReasoning:'
)

CALC = re.compile(r"<<[^>]*>>")

def clean_reasoning(answer: str) -> tuple[str, str]:
    # answer = reasoning ... "#### N"
    parts = answer.split("####")
    target = parts[-1].strip()
    reasoning = "####".join(parts[:-1])
    reasoning = CALC.sub("", reasoning)          # drop <<...>> calc annotations
    reasoning = re.sub(r"[ \t]+", " ", reasoning)
    reasoning = "\n".join(l.rstrip() for l in reasoning.splitlines()).strip()
    return reasoning, target

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open(args.template).read()

    ds = load_dataset("openai/gsm8k", "main", split="train")
    n = 0
    with open(args.out, "w") as f:
        for ex in ds:
            reasoning, target = clean_reasoning(ex["answer"])
            if not target or not reasoning:
                continue
            user = MATH_PROMPT_TEMPLATE.format(prompt=ex["question"].strip())
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
            completion = f"{reasoning}\n\nANSWER: {target}<end_of_turn>\n"
            f.write(json.dumps({"prompt": prompt, "completion": completion,
                                "target": target}) + "\n")
            n += 1
    print(f"wrote {n} rows to {args.out}")

if __name__ == "__main__":
    main()
