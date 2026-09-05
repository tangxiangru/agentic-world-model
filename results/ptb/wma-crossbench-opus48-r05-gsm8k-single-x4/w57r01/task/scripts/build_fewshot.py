#!/usr/bin/env python3
"""Build few-shot-structured SFT data so the model learns to STOP (emit
<end_of_turn>) after the FINAL question's answer even when the prompt contains
example Q/Reasoning/ANSWER blocks -- matching the grader's always-10-shot prompt.

- Target questions: GSM8K-train + a MetaMathQA GSM subset (reformatted).
- Few-shot context: k in {0,2,4,6,8,10} real GSM8K-train examples, rendered EXACTLY
  like inspect's sample_to_fewshot (raw reasoning WITH <<>>), placed in a system
  message -- the same shape the grader uses.
- Completion: clean reasoning + single 'ANSWER: N' + <end_of_turn> (loss on this only).
All source data is GSM8K TRAIN / MetaMathQA (train-derived); no test items.
"""
import argparse, json, re, random
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
    return s.strip().rstrip(".").replace("$", "").replace(",", "").strip()

def clean_gsm(answer):
    parts = answer.split("####")
    target = clean_num(parts[-1])
    r = CALC.sub("", "####".join(parts[:-1]))
    r = re.sub(r"[ \t]+", " ", r)
    r = "\n".join(l.rstrip() for l in r.splitlines()).strip()
    return r, target

def clean_meta(response):
    m = re.search(r"The answer is:\s*(.+?)\s*$", response.strip(), re.S)
    if not m:
        return None, None
    target = clean_num(m.group(1).splitlines()[0])
    body = CALC.sub("", response[:m.start()].split("####")[0])
    body = re.sub(r"[ \t]+", " ", body)
    body = "\n".join(l.rstrip() for l in body.splitlines()).strip()
    return body, target

def eval_fewshot(q, answer):
    # replicate inspect sample_to_fewshot: raw reasoning (keeps <<>>), then ANSWER
    a = answer.split("####")
    target = a[-1].strip()
    reasoning = "####".join(a[:-1]).strip()
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"

def valid(t):
    return bool(t) and bool(re.fullmatch(r"-?\d+(\.\d+)?", t))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_meta", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from datasets import load_dataset
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    ct = open(args.template).read()

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    fs_pool = [(ex["question"], ex["answer"]) for ex in gsm]

    # target examples: (question, clean_reasoning, target)
    targets = []
    for ex in gsm:
        r, t = clean_gsm(ex["answer"])
        if r and valid(t):
            targets.append((ex["question"], r, t))
    n_gsm = len(targets)

    meta = load_dataset("meta-math/MetaMathQA", split="train")
    pool = [ex for ex in meta if ex["type"] in ("GSM_Rephrased", "GSM_AnsAug")]
    rng.shuffle(pool)
    added = 0
    for ex in pool:
        if added >= args.n_meta:
            break
        r, t = clean_meta(ex["response"])
        if r and valid(t):
            targets.append((ex["query"], r, t))
            added += 1
    rng.shuffle(targets)

    kchoices = [0, 2, 4, 6, 8, 10]
    with open(args.out, "w") as f:
        for q, reasoning, target in targets:
            k = rng.choice(kchoices)
            msgs = []
            if k:
                shots = rng.sample(fs_pool, k + 2)
                shots = [s for s in shots if s[0] != q][:k]
                sysmsg = "\n\n".join(eval_fewshot(sq, sa) for sq, sa in shots)
                msgs.append({"role": "system", "content": sysmsg})
            msgs.append({"role": "user",
                         "content": MATH_PROMPT_TEMPLATE.format(prompt=q.strip())})
            prompt = tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True,
                                             chat_template=ct)
            completion = f"{reasoning}\n\nANSWER: {target}<end_of_turn>\n"
            f.write(json.dumps({"prompt": prompt, "completion": completion,
                                "target": target}) + "\n")
    print(f"gsm_targets={n_gsm} meta_added={added} total={len(targets)} -> {args.out}")

if __name__ == "__main__":
    main()
