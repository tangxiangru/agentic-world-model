#!/usr/bin/env python3
"""Combine rejection-sampled correct traces + GSM8K gold into an SFT set."""
import re, json, random, argparse
from datasets import load_dataset

random.seed(11)

def clean_gsm_reasoning(ans):
    parts = ans.split("####")
    reasoning = re.sub(r"<<[^>]*>>", "", parts[0].strip())
    target = parts[1].strip().replace(",", "").replace("$", "").strip()
    return reasoning, target

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rej", default="rej_data.jsonl")
    ap.add_argument("--out", default="train_star.jsonl")
    ap.add_argument("--gold_rep", type=int, default=1)
    ap.add_argument("--meta", type=int, default=0, help="num MetaMath GSM forward examples to add")
    args = ap.parse_args()

    out = []
    # gold GSM8K train
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for r in gsm:
        reasoning, target = clean_gsm_reasoning(r["answer"])
        for _ in range(args.gold_rep):
            out.append({"question": r["question"].strip(),
                        "response": f"{reasoning}\nANSWER: {target}", "src": "gold"})
    n_gold = len(out)

    # rejection-sampled
    n_rej = 0
    with open(args.rej) as f:
        for line in f:
            r = json.loads(line)
            out.append(r); n_rej += 1

    # optional MetaMath GSM forward (AnsAug + Rephrased only)
    n_meta = 0
    if args.meta > 0:
        meta = load_dataset("meta-math/MetaMathQA", split="train")
        ans_re = re.compile(r"The answer is:\s*(.+)\s*$")
        pool = []
        for r in meta:
            if r["type"] not in ("GSM_AnsAug", "GSM_Rephrased"):
                continue
            resp = r["response"].strip()
            m = ans_re.search(resp)
            if not m: continue
            num = m.group(1).strip().replace(",", "").replace("$", "").strip()
            if not re.fullmatch(r"-?\d+(\.\d+)?", num): continue
            idxs = [i for i in [resp.find("\n####"), resp.find("The answer is:")] if i != -1]
            cut = min(idxs) if idxs else len(resp)
            body = resp[:cut].strip()
            if len(body) < 10: continue
            pool.append({"question": r["query"].strip(),
                         "response": f"{body}\nANSWER: {num}", "src": "meta"})
        random.shuffle(pool)
        for r in pool[:args.meta]:
            out.append(r); n_meta += 1

    random.shuffle(out)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"gold={n_gold} rej={n_rej} meta={n_meta} TOTAL={len(out)}")

if __name__ == "__main__":
    main()
