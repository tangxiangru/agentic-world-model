#!/usr/bin/env python3
"""Build SFT data from the GSM8K TRAIN split, formatted to match the grader.

The grader (inspect_evals/gsm8k) wraps each question in MATH_PROMPT_TEMPLATE and
renders with templates/gemma3.jinja. Few-shot demos end each solution with
"ANSWER: <target>". We reproduce that target style for the model turn and end
the target with <end_of_turn> (token 106, an eos id in generation_config).

Output JSONL rows: {"prompt": <user turn content>, "completion": <model turn text>}
"""
import argparse, json, re
from datasets import load_dataset

# Byte-for-byte copy of inspect_evals.gsm8k.MATH_PROMPT_TEMPLATE (verified).
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def build(split_rows):
    rows = []
    for r in split_rows:
        q = r["question"].strip()
        ans = r["answer"]
        parts = ans.split("####")
        assert len(parts) == 2, ans
        reasoning = parts[0].strip()
        target = parts[1].strip()
        # target must be a clean number for the grader's numeric match
        target_clean = target.replace(",", "")
        prompt = MATH_PROMPT_TEMPLATE.format(prompt=q)
        # End the target with the stop token the grader stops on (id 106).
        completion = f"{reasoning}\n\nANSWER: {target_clean}<end_of_turn>"
        rows.append({"prompt": prompt, "completion": completion,
                     "target": target_clean})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gsm8k_train_sft.jsonl")
    args = ap.parse_args()
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = build(ds)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")
    # quick stats
    import statistics
    comp_lens = [len(x["completion"]) for x in rows]
    print("completion char len p50/max:", int(statistics.median(comp_lens)), max(comp_lens))
    print("--- example prompt ---")
    print(rows[0]["prompt"])
    print("--- example completion ---")
    print(rows[0]["completion"])


if __name__ == "__main__":
    main()
