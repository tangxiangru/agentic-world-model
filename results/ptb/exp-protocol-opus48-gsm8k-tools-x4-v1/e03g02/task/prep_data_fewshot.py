#!/usr/bin/env python3
"""Few-shot-aware GSM8K-train SFT data.

Mirrors the grader's structure: a system message holding k solved examples (in the
grader's exact fewshot format), then the user question wrapped in MATH_PROMPT_TEMPLATE.
Teaches the model to answer the LAST question once and stop (<end_of_turn>) even in a
multi-shot context -- fixing the exp-02 "answer-then-continue" losses.

Only the GSM8K TRAIN split is used (test forbidden). k varies to span the eval's 10-shot.
"""
import json, re, random, sys
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC = re.compile(r"<<[^>]*>>")

def clean(ans):
    parts = ans.split("####")
    reasoning = CALC.sub("", parts[0]).strip()
    final = parts[-1].replace(",", "").strip()
    return reasoning, final

def fewshot_block(q, reasoning, ans):
    # matches inspect_evals gsm8k sample_to_fewshot exactly (with clean reasoning)
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {ans}"

def main(out_path, seed=0):
    rng = random.Random(seed)
    ds = load_dataset("openai/gsm8k", "main", split="train")
    items = []
    for ex in ds:
        r, f = clean(ex["answer"])
        if not re.fullmatch(r"-?\d+(\.\d+)?", f):
            continue
        items.append((ex["question"].strip(), r, f))
    n = len(items)
    K_CHOICES = [2, 3, 4, 5, 6, 8, 10, 10]  # bias toward deeper contexts near the eval's 10-shot
    written = 0
    with open(out_path, "w") as fout:
        for i, (q, r, f) in enumerate(items):
            k = rng.choice(K_CHOICES)
            # sample k distinct other examples as fewshots
            idxs = rng.sample([j for j in range(n) if j != i], k)
            shots = "\n\n".join(fewshot_block(*items[j]) for j in idxs)
            rec = {
                "system": shots,
                "prompt": MATH_PROMPT_TEMPLATE.format(prompt=q),
                "completion": f"{r}\n\nANSWER: {f}",
                "answer": f,
                "k": k,
            }
            fout.write(json.dumps(rec) + "\n")
            written += 1
    print(f"wrote {written} few-shot examples to {out_path}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/gsm8k_train_fewshot.jsonl",
         int(sys.argv[2]) if len(sys.argv) > 2 else 0)
