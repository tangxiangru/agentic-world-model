#!/usr/bin/env python3
"""Prepare GSM8K *train* split into prompt/completion pairs matching the grader.

Grader (inspect_evals/gsm8k):
  - user content = MATH_PROMPT_TEMPLATE.format(prompt=question)
  - fewshot answer format: "{reasoning}\n\nANSWER: {target}"
  - scorer = match(numeric=True, location='end') -> extracts LAST number in output.

We train single-turn (no fewshot prefix): teach the model to emit reasoning then
"ANSWER: <int>" and stop with <end_of_turn>.  Only the GSM8K TRAIN split is used
(test is forbidden). Calculator <<...>> annotations are stripped; the '#### N'
line is replaced by 'ANSWER: N'.
"""
import json, re, sys
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC = re.compile(r"<<[^>]*>>")

def clean_reasoning(ans: str):
    parts = ans.split("####")
    reasoning = parts[0]
    final = parts[-1].strip()
    reasoning = CALC.sub("", reasoning).strip()
    # normalize gold number (strip commas)
    final_num = final.replace(",", "").strip()
    return reasoning, final_num

def main(out_path):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    n_written = 0
    with open(out_path, "w") as f:
        for ex in ds:
            q = ex["question"].strip()
            reasoning, final_num = clean_reasoning(ex["answer"])
            # sanity: final must be numeric
            if not re.fullmatch(r"-?\d+(\.\d+)?", final_num):
                continue
            prompt = MATH_PROMPT_TEMPLATE.format(prompt=q)
            completion = f"{reasoning}\n\nANSWER: {final_num}"
            rec = {"prompt": prompt, "completion": completion,
                   "question": q, "answer": final_num}
            f.write(json.dumps(rec) + "\n")
            n_written += 1
    print(f"wrote {n_written} examples to {out_path}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/gsm8k_train.jsonl")
