#!/usr/bin/env python3
"""CPU-only data prep. Build GSM8K-train SFT data in the exact eval format.

Eval (inspect_evals/gsm8k) renders, per item, a user turn:

  Solve the following math problem step by step. The last line of your
  response should be of the form "ANSWER: $ANSWER" ...

  {question}

  Remember to put your answer on its own line at the end in the form
  "ANSWER: $ANSWER" ... Reasoning:

then the gemma3.jinja template wraps it as
  <bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n

The scorer is match(numeric=True, location="end") -> it takes the LAST number
in the completion. So the target must end with "ANSWER: N" then stop at
<end_of_turn> (id 106, in the model's eos list).

We train ZERO-SHOT (single question, no fewshot exemplars). The eval adds a
fixed 10-shot system prefix at test time, which only reinforces the format.
"""
import json, re, sys
from pathlib import Path
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC = re.compile(r"<<[^>]*>>")


def render_prompt(question: str) -> str:
    content = MATH_PROMPT_TEMPLATE.format(prompt=question).strip()
    # gemma3.jinja zero-shot user turn, no system message
    return f"<bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"


def clean_reasoning(answer: str):
    parts = answer.split("####")
    reasoning = "####".join(parts[:-1])
    final = parts[-1].strip()
    final_num = final.replace(",", "").replace("$", "").strip()
    reasoning = CALC.sub("", reasoning)          # drop <<...>> calculator spans
    reasoning = re.sub(r"[ \t]+", " ", reasoning)  # squeeze spaces
    reasoning = reasoning.strip()
    return reasoning, final_num


def main():
    out_path = Path("data/gsm8k_train.jsonl")
    con_path = Path("data/contam_input.jsonl")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows, con = [], []
    for r in ds:
        q = r["question"].strip()
        reasoning, final = clean_reasoning(r["answer"])
        if not reasoning or not final:
            continue
        completion = f"{reasoning}\nANSWER: {final}<end_of_turn>"
        # exactly one ANSWER: marker
        assert completion.count("ANSWER:") == 1, completion
        rows.append({"prompt": render_prompt(q), "completion": completion, "answer": final})
        con.append({"question": q, "answer": f"{reasoning}\n#### {final}"})
    with out_path.open("w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    with con_path.open("w") as f:
        for x in con:
            f.write(json.dumps(x) + "\n")
    print(f"wrote {len(rows)} rows -> {out_path}")
    # show one rendered example
    print("=== EXAMPLE PROMPT ===")
    print(rows[0]["prompt"])
    print("=== EXAMPLE COMPLETION ===")
    print(rows[0]["completion"])


if __name__ == "__main__":
    main()
