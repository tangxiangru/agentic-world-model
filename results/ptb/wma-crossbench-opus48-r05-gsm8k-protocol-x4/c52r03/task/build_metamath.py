#!/usr/bin/env python3
"""Build SFT data from MetaMathQA GSM subset (derived from GSM8K TRAIN + MATH train; not test).
Reformats the double '#### N / The answer is: N' targets to a SINGLE 'ANSWER: N' marker.
Mixes with original GSM8K train. Writes JSONL {prompt, completion, gold, text}.
"""
import argparse, json, re, random

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC = re.compile(r"<<[^>]*>>")
ANS_IS = re.compile(r"The answer is:\s*(.+?)\s*$", re.MULTILINE)
HASH = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)

def num_ok(s):
    s = s.strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "")
    try:
        float(s); return s
    except Exception:
        return None

def reformat(resp):
    # find gold: prefer 'The answer is: X', fallback '#### X'
    gold = None
    m = list(ANS_IS.finditer(resp))
    if m:
        gold = m[-1].group(1)
    else:
        m2 = list(HASH.finditer(resp))
        if m2:
            gold = m2[-1].group(1)
    if gold is None:
        return None, None
    gold_clean = num_ok(gold)
    if gold_clean is None:
        return None, None
    # strip markers and calc spans from body
    body = ANS_IS.sub("", resp)
    body = HASH.sub("", body)
    body = CALC.sub("", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if not body:
        return None, None
    completion = f"{body}\nANSWER: {gold_clean}"
    return completion, gold_clean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", nargs="+", default=["GSM_AnsAug", "GSM_Rephrased"])
    ap.add_argument("--cap_per_type", type=int, default=40000)
    ap.add_argument("--include_gsm8k_train", default="data/gsm8k_train.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    rng = random.Random(args.seed)
    by_type = {t: [] for t in args.types}
    for r in ds:
        t = r["type"]
        if t in by_type:
            by_type[t].append(r)
    rows = []
    kept = dropped = 0
    for t in args.types:
        items = by_type[t]
        rng.shuffle(items)
        n = 0
        for r in items:
            if n >= args.cap_per_type:
                break
            comp, gold = reformat(r["response"])
            if comp is None:
                dropped += 1; continue
            q = r["query"].strip()
            prompt = MATH_PROMPT_TEMPLATE.format(prompt=q)
            rows.append({"prompt": prompt, "completion": comp, "gold": gold,
                         "text": prompt + "\n" + comp})
            n += 1; kept += 1
        print(f"  {t}: kept {n}")
    # include original gsm8k train
    if args.include_gsm8k_train:
        for line in open(args.include_gsm8k_train):
            rows.append(json.loads(line))
        print(f"  original gsm8k_train added")
    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} (metamath kept={kept}, dropped={dropped})")

if __name__ == "__main__":
    main()
