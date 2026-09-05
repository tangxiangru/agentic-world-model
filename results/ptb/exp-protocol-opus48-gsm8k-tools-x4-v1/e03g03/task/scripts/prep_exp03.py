#!/usr/bin/env python3
"""Build augmented SFT set: GSM8K-train (clean) + MetaMathQA GSM subset.

MetaMathQA GSM responses end with '#### N\\nThe answer is: N' (double marker).
We cut at the first marker, strip <<...>> spans, and append a single
'ANSWER: N' + <end_of_turn> so the grader's last-number rule reads N exactly.
"""
import json, re, random
from pathlib import Path
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC = re.compile(r"<<[^>]*>>")
ANS_RE = re.compile(r"The answer is:\s*(.+?)\s*$", re.MULTILINE)
N_META = 40000
SEED = 0


def render_prompt(question):
    content = MATH_PROMPT_TEMPLATE.format(prompt=question.strip()).strip()
    return f"<bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"


def norm_ans(a):
    a = a.strip().strip(".").replace(",", "").replace("$", "").strip()
    return a


def meta_rows():
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    ds = ds.filter(lambda r: r["type"] in ("GSM_Rephrased", "GSM_AnsAug"))
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    out = []
    for i in idx:
        r = ds[i]
        resp = r["response"]
        m = ANS_RE.search(resp)
        if not m:
            continue
        ans = norm_ans(m.group(1))
        if not re.fullmatch(r"-?\d+(\.\d+)?", ans):
            continue
        # cut reasoning at first marker
        cut = len(resp)
        for mark in ("####", "The answer is:"):
            j = resp.find(mark)
            if j != -1:
                cut = min(cut, j)
        reasoning = CALC.sub("", resp[:cut])
        reasoning = re.sub(r"[ \t]+", " ", reasoning).strip()
        if not reasoning:
            continue
        completion = f"{reasoning}\nANSWER: {ans}<end_of_turn>"
        if completion.count("ANSWER:") != 1:
            continue
        out.append({"prompt": render_prompt(r["query"]), "completion": completion,
                    "answer": ans, "src": "metamath"})
        if len(out) >= N_META:
            break
    return out


def main():
    random.seed(SEED)
    gsm = [json.loads(l) for l in open("data/gsm8k_train.jsonl")]
    for g in gsm:
        g["src"] = "gsm8k_train"
    meta = meta_rows()
    print(f"gsm8k_train={len(gsm)} metamath={len(meta)}")
    rows = gsm + meta
    random.Random(SEED).shuffle(rows)
    with open("data/exp03_train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # contamination input (question+answer text)
    with open("data/exp03_contam.jsonl", "w") as f:
        for r in meta:  # gsm8k_train already checked in exp-02
            # recover question text from prompt
            q = r["prompt"].split("step by step.")[-1]
            f.write(json.dumps({"question": q, "answer": r["completion"]}) + "\n")
    print(f"wrote {len(rows)} rows -> data/exp03_train.jsonl")
    print("EXAMPLE metamath completion:\n", meta[0]["completion"][:400])


if __name__ == "__main__":
    main()
