#!/usr/bin/env python3
"""Build GSM8K-train SFT data in the exact eval per-question format.

Grader (inspect_evals/gsm8k) uses, per question:
  user prompt = MATH_PROMPT_TEMPLATE.format(prompt=question)
  gold answer format shown in few-shot = "<reasoning>\n\nANSWER: <target>"
  scorer = match(numeric=True, location='end')  -> reads LAST numeric token.

We reproduce this: target = clean CoT (calculator <<...>> stripped) + "\n\nANSWER: N".
Source is the GSM8K TRAIN split only (disjoint from the test set).
"""
import json, re, os

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
    target = parts[-1].strip()
    reasoning = "####".join(parts[:-1]).strip()
    reasoning = CALC.sub("", reasoning)          # remove <<48/2=24>> calculator markers
    reasoning = re.sub(r"[ \t]+", " ", reasoning)  # collapse spaces created by removal
    reasoning = re.sub(r" *\n *", "\n", reasoning).strip()
    return reasoning, target

def main():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    out_msgs = open("data/train_messages.jsonl", "w")
    out_text = open("data/train_text.jsonl", "w")  # for contamination check (one doc/line)
    n = 0
    for r in ds:
        q = r["question"].strip()
        reasoning, target = clean_reasoning(r["answer"])
        if not reasoning or not target:
            continue
        user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        assistant = f"{reasoning}\n\nANSWER: {target}"
        out_msgs.write(json.dumps({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            # `completion` mirrors the true training target: what the chat template renders
            # after the generation prompt, ending in the stop token <end_of_turn> (id 106).
            "completion": assistant + "<end_of_turn>",
            "gold": target,
        }) + "\n")
        # contamination doc: question + full answer text (what we actually train on)
        out_text.write(json.dumps({"text": q + "\n" + assistant}) + "\n")
        n += 1
    out_msgs.close(); out_text.close()
    print(f"wrote {n} examples")

if __name__ == "__main__":
    main()
