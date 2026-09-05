#!/usr/bin/env python3
"""Build few-shot-CONTEXT SFT data to teach stop-after-one-answer under the eval's
10-shot prompt. Root cause (exp-02): the model, shown a long in-context list of
problems, continues the list past its own answer instead of emitting <end_of_turn>.

Each row = optional system message of k GSM8K-train exemplars in the grader's EXACT
few-shot format (question + Reasoning + raw reasoning + 'ANSWER: N'), then the target
question, then the clean target completion ending in <end_of_turn>. k varies so the
model learns to stop regardless of how many prior problems it has seen.
All exemplars/targets are GSM8K TRAIN (disjoint from test).
"""
import json, re, random

from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC = re.compile(r"<<[^>]*>>")
K_CHOICES = [0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 8]  # weighted; covers zero-shot..high-shot
PASSES = 2  # two independent few-shot samplings per question -> more stopping signal


def clean(ans):
    parts = ans.split("####")
    target = parts[-1].strip().replace("$", "").replace(",", "")
    reasoning = "####".join(parts[:-1]).strip()
    reasoning = CALC.sub("", reasoning)
    reasoning = re.sub(r"[ \t]+", " ", reasoning)
    reasoning = re.sub(r" *\n *", "\n", reasoning).strip()
    return reasoning, target


def raw_fewshot(q, raw_answer):
    # grader's exact few-shot rendering: keeps raw reasoning (with <<>> and no ####)
    a = raw_answer.split("####")
    target = a.pop().strip()
    reasoning = "####".join(a).strip()
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def main():
    random.seed(0)
    ds = load_dataset("openai/gsm8k", "main", split="train")
    items = []
    for r in ds:
        reasoning, target = clean(r["answer"])
        if reasoning and re.fullmatch(r"-?\d+(\.\d+)?", target):
            items.append({"q": r["question"].strip(), "reasoning": reasoning,
                          "target": target, "raw": r["answer"]})
    n = len(items)
    out_msgs = open("data/train_fewshot_messages.jsonl", "w")
    out_text = open("data/train_fewshot_text.jsonl", "w")
    written = 0
    for _ in range(PASSES):
        for i, it in enumerate(items):
            k = random.choice(K_CHOICES)
            msgs = []
            if k > 0:
                pool = random.sample(range(n), k + 1)
                pool = [j for j in pool if j != i][:k]
                shots = [raw_fewshot(items[j]["q"], items[j]["raw"]) for j in pool]
                msgs.append({"role": "system", "content": "\n\n".join(shots)})
            assistant = f'{it["reasoning"]}\n\nANSWER: {it["target"]}'
            msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=it["q"])})
            msgs.append({"role": "assistant", "content": assistant})
            out_msgs.write(json.dumps({"messages": msgs, "completion": assistant + "<end_of_turn>"}) + "\n")
            out_text.write(json.dumps({"text": it["q"] + "\n" + assistant}) + "\n")
            written += 1
    out_msgs.close(); out_text.close()
    print(f"wrote {written} rows from {n} questions x {PASSES} passes")


if __name__ == "__main__":
    main()
