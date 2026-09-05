#!/usr/bin/env python3
"""Convert MetaMathQA GSM-derived items (built from GSM8K *train*) into eval-format SFT data."""
import argparse, json, random, re
from prep_data import MATH_PROMPT_TEMPLATE, clean_number, sample_to_fewshot
from datasets import load_dataset

PATH = "/home/ben/hf_cache/hub/datasets--meta-math--MetaMathQA/snapshots/aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18/MetaMathQA-395K.json"
BAD = ("```", "\\begin{", "http", "<<", "\\boxed", "\\[", "\\frac", "$\\")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/metamath.jsonl")
    ap.add_argument("--types", default="GSM_Rephrased,GSM_AnsAug")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    args = ap.parse_args()
    types = set(args.types.split(","))
    rng = random.Random(3)

    data = json.load(open(PATH))
    rows = [d for d in data if d["type"] in types]
    rng.shuffle(rows)

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    fewshot_pool = [(r["question"], "####".join(r["answer"].split("####")[:-1]).strip(),
                     r["answer"].split("####")[-1].strip()) for r in gsm]

    out, seen = [], set()
    for d in rows:
        if len(out) >= args.n:
            break
        resp = d["response"]
        m = re.search(r"The answer is:\s*(.+)\s*$", resp.strip())
        if not m:
            continue
        ans = clean_number(m.group(1))
        if ans is None:
            continue
        body = resp.split("The answer is:")[0]
        body = re.sub(r"####.*", "", body).strip()
        if len(body) < 30 or len(body) > 2500:
            continue
        if any(b in body for b in BAD):
            continue
        q = d["query"].strip()
        key = (q, ans)
        if key in seen:
            continue
        seen.add(key)
        out.append((q, f"{body}\n\nANSWER: {ans}"))

    print("kept", len(out))
    with open(args.out, "w") as fh, open(args.out + ".decon", "w") as fd:
        for q, t in out:
            msgs = []
            if rng.random() < args.fewshot_frac:
                k = rng.choices([2, 4, 10], weights=[0.4, 0.3, 0.3])[0]
                msgs.append({"role": "system", "content": "\n\n".join(
                    sample_to_fewshot(*s) for s in rng.sample(fewshot_pool, k))})
            msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)})
            msgs.append({"role": "assistant", "content": t})
            fh.write(json.dumps({"messages": msgs, "source": "metamath"}) + "\n")
            fd.write(json.dumps({"text": q + "\n" + t}) + "\n")


if __name__ == "__main__":
    main()
