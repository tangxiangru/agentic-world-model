#!/usr/bin/env python3
"""Build SFT data for GSM8K, rendered byte-identically to the grader's template.

Grader facts (inspect_evals/gsm8k/gsm8k.py + templates/gemma3.jinja):
  - user turn  = MATH_PROMPT_TEMPLATE.format(prompt=question)
  - assistant  = step-by-step reasoning, last line "ANSWER: $ANSWER"
  - scorer     = match(location="end", numeric=True) -> answer must be LAST
  - stop token = <end_of_turn>
"""
import json
import re
import argparse
from datasets import load_dataset

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

# copied verbatim from inspect_evals.gsm8k.gsm8k.MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC = re.compile(r"<<[^>]*>>")


def clean_solution(ans: str):
    """Strip calculator annotations and the '#### N' marker; return (reasoning, target)."""
    body, _, target = ans.partition("####")
    target = target.strip()
    body = CALC.sub("", body).strip()
    return body, target


def build(split="train"):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = []
    for r in ds:
        reasoning, target = clean_solution(r["answer"])
        if not target:
            continue
        # exactly one answer marker, and it is the final line
        completion = f"{reasoning}\n\nANSWER: {target}"
        assert completion.count("####") == 0
        assert completion.strip().endswith(f"ANSWER: {target}")
        rows.append(
            {
                "prompt": MATH_PROMPT_TEMPLATE.format(prompt=r["question"].strip()),
                "completion": completion,
                "target": target,
            }
        )
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="train_data.jsonl")
    a = ap.parse_args()

    rows = build("train")
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {a.out}")

    # ---- preflight: render one example the way training will, and count tokens ----
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAP)
    with open("templates/gemma3.jinja") as f:
        tok.chat_template = f.read()

    ex = rows[0]
    prompt_txt = tok.apply_chat_template(
        [{"role": "user", "content": ex["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_txt = prompt_txt + ex["completion"] + "<end_of_turn>\n"
    print("=" * 30, "RENDERED EXAMPLE", "=" * 30)
    print(repr(full_txt))
    print("=" * 78)

    lens = []
    for r in rows:
        p = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        lens.append(len(tok(p + r["completion"] + "<end_of_turn>\n", add_special_tokens=False)["input_ids"]))
    lens.sort()
    print(f"token len p50={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} max={lens[-1]}")
