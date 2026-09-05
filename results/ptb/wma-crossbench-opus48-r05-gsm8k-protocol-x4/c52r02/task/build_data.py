#!/usr/bin/env python3
"""Build GSM8K-train SFT data matching the evaluate.py per-question format.

Eval per-question format (from inspect_evals/gsm8k):
  user prompt = MATH_PROMPT_TEMPLATE.format(prompt=question)  (ends with 'Reasoning:')
  assistant   = '{reasoning}\n\nANSWER: {target}'
Few-shot examples use exactly the same shape. We train zero-shot on this shape.
Reasoning = GSM8K answer text before the final '#### N' line (calculator
annotations <<...>> stripped). Single answer marker: 'ANSWER: N'.
"""
import argparse, json, re
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

def clean_reasoning(answer: str, keep_calc: bool = False):
    parts = answer.split("####")
    target = parts[-1].strip()
    reasoning = "####".join(parts[:-1]).strip()
    if not keep_calc:
        reasoning = re.sub(r"<<[^>]*>>", "", reasoning)  # strip calculator annotations
    return reasoning, target

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gsm8k_train.jsonl")
    ap.add_argument("--keep_calc", action="store_true",
                    help="keep <<a=b>> calculator annotations (match the 10-shot eval format)")
    args = ap.parse_args()
    ds = load_dataset("openai/gsm8k", "main", split="train")
    n = 0
    with open(args.out, "w") as f:
        for r in ds:
            reasoning, target = clean_reasoning(r["answer"], keep_calc=args.keep_calc)
            assert target.replace(",", "").lstrip("-").isdigit() or re.match(r"^-?[\d,]+(\.\d+)?$", target), target
            prompt = MATH_PROMPT_TEMPLATE.format(prompt=r["question"].strip())
            completion = f"{reasoning}\n\nANSWER: {target}<end_of_turn>"
            f.write(json.dumps({"prompt": prompt, "completion": completion,
                                "question": r["question"].strip(), "target": target}) + "\n")
            n += 1
    print(f"wrote {n} examples to {args.out}")

if __name__ == "__main__":
    main()
