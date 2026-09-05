#!/usr/bin/env python3
"""Build SFT data from GSM8K train split, matching the eval prompt format."""
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
    """Split GSM8K answer into (reasoning, final_answer). Strip <<calc>> annotations."""
    parts = answer.split("####")
    final = parts[-1].strip()
    reasoning = "####".join(parts[:-1]).strip()
    # remove calculator annotations like <<48/2=24>>
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning)
    return reasoning, final


def main():
    ds = load_dataset("openai/gsm8k", "main")["train"]
    out = []
    for rec in ds:
        q = rec["question"].strip()
        reasoning, final = clean_reasoning(rec["answer"])
        # normalize final: strip commas
        final_clean = final.replace(",", "")
        user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        completion = f"{reasoning}\n\nANSWER: {final_clean}"
        out.append({"prompt": user, "completion": completion,
                    "question": q, "answer": final_clean})
    with open("train_sft.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    # decontamination input: full text of each doc
    with open("decon_input.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps({"text": r["question"] + " " + r["completion"]}) + "\n")
    print(f"wrote {len(out)} examples")
    print("=== sample ===")
    print(out[0]["prompt"])
    print("--- completion ---")
    print(out[0]["completion"])


if __name__ == "__main__":
    main()
