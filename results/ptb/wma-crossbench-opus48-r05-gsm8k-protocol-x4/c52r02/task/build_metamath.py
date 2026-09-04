#!/usr/bin/env python3
"""Build exp-02 SFT data: MetaMathQA GSM subset (derived from GSM8K TRAIN),
reformatted to a SINGLE 'ANSWER: N' marker + <end_of_turn>, plus the original
GSM8K-train file. Prompts wrapped in the eval's MATH_PROMPT_TEMPLATE."""
import argparse, json, re, random
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

def reformat(query, response):
    if "####" not in response:
        return None
    head, tail = response.split("####", 1)
    ans = tail.strip().split("\n")[0].strip()
    ans = ans.replace(",", "").replace("$", "").strip()
    if not re.match(r"^-?\d+(\.\d+)?$", ans):
        return None
    reasoning = re.sub(r"<<[^>]*>>", "", head).strip()
    # drop any stray 'The answer is' inside reasoning
    reasoning = re.sub(r"\n?The answer is:.*$", "", reasoning).strip()
    if not reasoning:
        return None
    prompt = MATH_PROMPT_TEMPLATE.format(prompt=query.strip())
    completion = f"{reasoning}\n\nANSWER: {ans}<end_of_turn>"
    return {"prompt": prompt, "completion": completion, "target": ans}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_meta", type=int, default=80000)
    ap.add_argument("--out", default="data/exp02_train.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    gsm_idx = [i for i, t in enumerate(ds["type"]) if t.startswith("GSM")]
    print("total GSM examples:", len(gsm_idx))
    random.shuffle(gsm_idx)
    gsm_idx = gsm_idx[:args.n_meta]
    rows = []
    dropped = 0
    for i in gsm_idx:
        r = ds[i]
        out = reformat(r["query"], r["response"])
        if out is None:
            dropped += 1; continue
        rows.append(out)
    print(f"metamath kept {len(rows)}, dropped {dropped}")
    # add original gsm8k train
    n_orig = 0
    for l in open("data/gsm8k_train.jsonl"):
        d = json.loads(l)
        rows.append({"prompt": d["prompt"], "completion": d["completion"], "target": d["target"]})
        n_orig += 1
    print("added original gsm8k train:", n_orig)
    random.shuffle(rows)
    with open(args.out, "w") as f:
        for d in rows:
            f.write(json.dumps(d) + "\n")
    print(f"wrote {len(rows)} to {args.out}")

if __name__ == "__main__":
    main()
