#!/usr/bin/env python3
"""Build SFT data that mimics the eval's few-shot prompt structure so the model
learns to STOP (emit <end_of_turn>) right after its own ANSWER line, even when
preceding in-context examples show 'ANSWER: X\\n\\n<next question>' patterns.

Each example: user turn = K few-shot blocks (raw GSM8K style, matching
sample_to_fewshot) + MATH_PROMPT_TEMPLATE(target_question); response = target
reasoning + 'ANSWER: X'. A fraction are zero-shot for robustness.
"""
import re, json, random, argparse
from datasets import load_dataset

random.seed(23)

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

def clean_gsm(ans):
    parts = ans.split("####")
    reasoning = re.sub(r"<<[^>]*>>", "", parts[0].strip())
    target = parts[1].strip().replace(",", "").replace("$", "").strip()
    return reasoning, target

def raw_gsm(ans):
    # keep raw reasoning (with <<>>) as eval's sample_to_fewshot does
    parts = ans.split("####")
    return parts[0].strip(), parts[1].strip().replace(",", "").replace("$", "").strip()

def fewshot_block(q, raw_reasoning, target):
    return f"{q}\n\nReasoning:\n{raw_reasoning}\n\nANSWER: {target}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rej", default="rej_data.jsonl")
    ap.add_argument("--out", default="train_fs.jsonl")
    ap.add_argument("--min_shots", type=int, default=2)
    ap.add_argument("--max_shots", type=int, default=5)
    ap.add_argument("--zeroshot_frac", type=float, default=0.25)
    args = ap.parse_args()

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    # few-shot pool (raw reasoning + question + target)
    pool = []
    for r in gsm:
        rr, tgt = raw_gsm(r["answer"])
        pool.append((r["question"].strip(), rr, tgt))

    # targets: gold (clean) + rejection-sampled
    targets = []
    for r in gsm:
        reasoning, target = clean_gsm(r["answer"])
        targets.append({"question": r["question"].strip(),
                        "response": f"{reasoning}\nANSWER: {target}"})
    with open(args.rej) as f:
        for line in f:
            r = json.loads(line)
            targets.append({"question": r["question"].strip(), "response": r["response"]})

    random.shuffle(targets)
    npool = len(pool)
    out = []
    for t in targets:
        q = t["question"]
        if random.random() < args.zeroshot_frac:
            user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        else:
            k = random.randint(args.min_shots, args.max_shots)
            blocks = []
            tries = 0
            while len(blocks) < k and tries < k * 4:
                tries += 1
                pq, prr, ptgt = pool[random.randrange(npool)]
                if pq == q:  # avoid leaking the exact target question as a shot
                    continue
                blocks.append(fewshot_block(pq, prr, ptgt))
            prefix = "\n\n".join(blocks)
            user = prefix + "\n\n" + MATH_PROMPT_TEMPLATE.format(prompt=q)
        out.append({"user": user, "response": t["response"], "src": "fs"})

    random.shuffle(out)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"targets={len(targets)} out={len(out)} zeroshot~{args.zeroshot_frac}")

if __name__ == "__main__":
    main()
