#!/usr/bin/env python3
"""Build an augmented SFT mixture: original GSM8K-train CoT + a subsample of
MetaMathQA GSM entries (derived from GSM8K TRAIN + rephrasings), reformatted to
a single 'ANSWER: N' marker ending in <end_of_turn>. No test-derived data.
"""
import argparse, json, re
MATH_PROMPT_TEMPLATE = (
    'Solve the following math problem step by step. The last line of your '
    'response should be of the form "ANSWER: $ANSWER" (without quotes) where '
    '$ANSWER is the answer to the problem.\n\n{prompt}\n\nRemember to put your '
    'answer on its own line at the end in the form "ANSWER: $ANSWER" (without '
    'quotes) where $ANSWER is the answer to the problem, and you do not need to '
    'use a \\boxed command.\n\nReasoning:'
)
CALC = re.compile(r"<<[^>]*>>")

def clean_num(s):
    s = s.strip().rstrip(".").replace("$", "").replace(",", "").strip()
    return s

def clean_reasoning_gsm(answer):
    parts = answer.split("####")
    target = clean_num(parts[-1])
    reasoning = CALC.sub("", "####".join(parts[:-1]))
    reasoning = re.sub(r"[ \t]+", " ", reasoning)
    reasoning = "\n".join(l.rstrip() for l in reasoning.splitlines()).strip()
    return reasoning, target

def clean_meta(response):
    # response: reasoning ... [maybe "#### N"] "The answer is: N"
    m = re.search(r"The answer is:\s*(.+?)\s*$", response.strip(), re.S)
    if not m:
        return None, None
    target = clean_num(m.group(1).splitlines()[0])
    body = response[:m.start()]
    body = body.split("####")[0]            # drop trailing '#### N' if present
    body = CALC.sub("", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = "\n".join(l.rstrip() for l in body.splitlines()).strip()
    return body, target

def render(tok, question, reasoning, target):
    user = MATH_PROMPT_TEMPLATE.format(prompt=question.strip())
    prompt = tok.apply_chat_template([{"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=True)
    completion = f"{reasoning}\n\nANSWER: {target}<end_of_turn>\n"
    return {"prompt": prompt, "completion": completion, "target": target}

def valid(target):
    return bool(target) and bool(re.fullmatch(r"-?\d+(\.\d+)?", target))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_rephrased", type=int, default=25000)
    ap.add_argument("--n_ansaug", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from datasets import load_dataset
    import random
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open(args.template).read()

    rows = []
    # 1. original GSM8K train
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for ex in gsm:
        r, t = clean_reasoning_gsm(ex["answer"])
        if r and valid(t):
            rows.append(render(tok, ex["question"], r, t))
    n_orig = len(rows)

    # 2. MetaMathQA GSM subsets
    meta = load_dataset("meta-math/MetaMathQA", split="train")
    by_type = {"GSM_Rephrased": [], "GSM_AnsAug": []}
    for ex in meta:
        if ex["type"] in by_type:
            by_type[ex["type"]].append(ex)
    for typ, n in [("GSM_Rephrased", args.n_rephrased), ("GSM_AnsAug", args.n_ansaug)]:
        pool = by_type[typ]
        rng.shuffle(pool)
        added = 0
        for ex in pool:
            if added >= n:
                break
            r, t = clean_meta(ex["response"])
            if r and valid(t):
                rows.append(render(tok, ex["query"], r, t))
                added += 1
        print(f"{typ}: added {added}")
    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"orig gsm={n_orig}; total={len(rows)} -> {args.out}")

if __name__ == "__main__":
    main()
