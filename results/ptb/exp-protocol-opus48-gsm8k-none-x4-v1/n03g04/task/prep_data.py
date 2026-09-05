#!/usr/bin/env python3
"""Build SFT training data for GSM8K, formatted to match the eval prompt/target.

Output: train_data.jsonl with fields {"question", "reasoning_answer"} where
reasoning_answer is the assistant target ending in a line "ANSWER: N".
Also emits train_text.jsonl (plain concatenation) for contamination checking.
"""
import json
import re
import random

random.seed(0)

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def clean_gsm8k_reasoning(ans: str):
    """GSM8K train answer -> (reasoning_text, final_answer)."""
    parts = ans.split("####")
    reasoning = parts[0].strip()
    final = parts[1].strip() if len(parts) > 1 else ""
    # strip calculator annotations <<...>>
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning)
    # normalize final number: remove commas, $, spaces
    final = final.replace(",", "").replace("$", "").strip()
    return reasoning, final


def norm_num(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    return s


def build_target(reasoning: str, final: str) -> str:
    reasoning = reasoning.strip()
    return f"{reasoning}\nANSWER: {final}"


def main():
    from datasets import load_dataset

    examples = []

    # 1) GSM8K train
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for r in gsm:
        reasoning, final = clean_gsm8k_reasoning(r["answer"])
        if not final:
            continue
        examples.append({"question": r["question"].strip(),
                         "reasoning_answer": build_target(reasoning, final),
                         "src": "gsm8k"})
    print("gsm8k train:", len(examples))

    # 2) MetaMathQA (GSM-derived types). Reformat "The answer is: X" -> ANSWER: X
    mm = load_dataset("meta-math/MetaMathQA", split="train")
    gsm_types = {"GSM_AnsAug", "GSM_Rephrased", "GSM_FOBAR", "GSM_SV"}
    ans_re = re.compile(r"The answer is:\s*(.*?)\s*$", re.DOTALL)
    mm_count = 0
    mm_examples = []
    for r in mm:
        if r["type"] not in gsm_types:
            continue
        resp = r["response"]
        m = ans_re.search(resp.strip())
        if not m:
            continue
        final = norm_num(m.group(1).split("\n")[0])
        # keep only clean numeric answers (GSM answers are integers/decimals)
        if not re.fullmatch(r"-?\d+(\.\d+)?", final):
            continue
        # reasoning = response with the trailing "The answer is:" line removed
        reasoning = resp[:m.start()].strip()
        # some responses have boxed; strip \boxed{...}
        reasoning = re.sub(r"\\boxed\{([^}]*)\}", r"\1", reasoning)
        if len(reasoning) < 10:
            continue
        mm_examples.append({"question": r["query"].strip(),
                            "reasoning_answer": build_target(reasoning, final),
                            "src": "metamath_gsm"})
        mm_count += 1
    print("metamath gsm-derived:", mm_count)

    random.shuffle(mm_examples)
    # cap metamath to keep training time reasonable and balanced
    MM_CAP = 30000
    mm_examples = mm_examples[:MM_CAP]
    examples.extend(mm_examples)

    random.shuffle(examples)
    with open("train_data.jsonl", "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print("total examples:", len(examples))

    # text file for contamination check (question + reasoning)
    with open("train_text.jsonl", "w") as f:
        for e in examples:
            doc = e["question"] + "\n" + e["reasoning_answer"]
            f.write(json.dumps(doc) + "\n")


if __name__ == "__main__":
    main()
