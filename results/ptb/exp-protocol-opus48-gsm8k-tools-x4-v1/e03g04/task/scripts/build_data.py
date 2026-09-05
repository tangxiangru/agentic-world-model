#!/usr/bin/env python3
"""Build SFT data from the GSM8K *train* split (not test => no contamination).

Each example is rendered to match the grader's per-item structure:
  user    = MATH_PROMPT_TEMPLATE.format(prompt=question)   (ends with "Reasoning:")
  assistant = <cleaned reasoning> + "\n\nANSWER: <N>"

We store raw fields (question, reasoning, answer) as JSONL so the trainer can
apply the gemma chat template itself. We also emit a plain-text JSONL for the
contamination checker.
"""
import json
import re
import argparse
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC = re.compile(r"<<[^>]*>>")


def clean_reasoning(ans: str) -> tuple[str, str]:
    parts = ans.split("####")
    reasoning = "####".join(parts[:-1]).strip()
    target = parts[-1].strip().replace(",", "")
    reasoning = CALC.sub("", reasoning)          # drop <<...>> calculator tags
    reasoning = re.sub(r"[ \t]+", " ", reasoning)
    reasoning = re.sub(r"\n{3,}", "\n\n", reasoning).strip()
    return reasoning, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gsm8k_train_sft.jsonl")
    ap.add_argument("--text-out", default="data/gsm8k_train_text.jsonl")
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split="train")
    n = 0
    with open(args.out, "w") as f, open(args.text_out, "w") as ft:
        for rec in ds:
            q = rec["question"].strip()
            reasoning, target = clean_reasoning(rec["answer"])
            if not target:
                continue
            user = MATH_PROMPT_TEMPLATE.format(prompt=q)
            assistant = f"{reasoning}\n\nANSWER: {target}<end_of_turn>"
            f.write(json.dumps({
                "question": q,
                "prompt": user,
                "completion": assistant,
                "answer": target,
            }) + "\n")
            # text for contamination checker: the full document we train on
            ft.write(json.dumps({"text": user + "\n" + assistant}) + "\n")
            n += 1
    print(f"wrote {n} examples to {args.out}")


if __name__ == "__main__":
    main()
