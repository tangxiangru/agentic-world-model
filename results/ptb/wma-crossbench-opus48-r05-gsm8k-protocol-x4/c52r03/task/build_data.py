#!/usr/bin/env python3
"""Build SFT data for GSM8K from the GSM8K TRAIN split (allowed; not test).
Cleans calculator annotations and rewrites '#### N' as 'ANSWER: N' (single marker).
Outputs JSONL with {prompt, completion, gold, text}.
"""
import argparse, json, re

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC = re.compile(r"<<[^>]*>>")

def clean_gsm8k_answer(ans: str):
    delim = "####"
    parts = ans.split(delim)
    gold = parts[-1].strip()
    reasoning = delim.join(parts[:-1]).strip()
    reasoning = CALC.sub("", reasoning)
    reasoning = re.sub(r"[ \t]+", " ", reasoning)
    reasoning = re.sub(r"\n{3,}", "\n\n", reasoning).strip()
    completion = f"{reasoning}\nANSWER: {gold}"
    return completion, gold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    n = 0
    with open(args.out, "w") as f:
        for r in ds:
            q = r["question"].strip()
            completion, gold = clean_gsm8k_answer(r["answer"])
            prompt = MATH_PROMPT_TEMPLATE.format(prompt=q)
            text = prompt + "\n" + completion
            f.write(json.dumps({"prompt": prompt, "completion": completion,
                                "gold": gold, "text": text}) + "\n")
            n += 1
    print(f"wrote {n} examples to {args.out}")

if __name__ == "__main__":
    main()
