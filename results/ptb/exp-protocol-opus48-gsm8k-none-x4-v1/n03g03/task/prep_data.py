#!/usr/bin/env python3
"""Build SFT training data from GSM8K train split, formatted to match the eval."""
import json
import re
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def clean_reasoning(answer: str):
    """Split gsm8k answer into (reasoning, final_answer), removing <<>> calc annotations."""
    DELIM = "####"
    parts = answer.split(DELIM)
    target = parts[-1].strip()
    reasoning = DELIM.join(parts[:-1]).strip()
    # remove calculator annotations like <<48/2=24>>
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning)
    # normalize target (strip commas)
    target = target.replace(",", "")
    return reasoning, target


def main():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for rec in ds:
        q = rec["question"].strip()
        reasoning, target = clean_reasoning(rec["answer"])
        completion = f"{reasoning}\n\nANSWER: {target}"
        user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        rows.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": completion},
            ],
            # also store plain question/answer for contamination checking / rejection sampling
            "question": q,
            "target": target,
        })
    with open("train_gsm8k.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} examples to train_gsm8k.jsonl")
    # print a sample
    print("\n=== SAMPLE ===")
    print(rows[0]["messages"][0]["content"])
    print("--- assistant ---")
    print(rows[0]["messages"][1]["content"])


if __name__ == "__main__":
    main()
