#!/usr/bin/env python3
import json, re, random
random.seed(3)

def clean_gsm8k(ans):
    parts = ans.split("####")
    reasoning = re.sub(r"<<[^>]*>>", "", parts[0]).strip()
    final = parts[1].strip().replace(",", "").replace("$", "") if len(parts) > 1 else ""
    return reasoning, final

def norm_num(s):
    return s.strip().replace(",", "").replace("$", "").rstrip(".")

def main():
    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="train")

    fewshot_pool = []
    for r in gsm:
        reasoning_raw = r["answer"].split("####")[0].strip()
        _, final = clean_gsm8k(r["answer"])
        if final:
            fewshot_pool.append(f"{r['question'].strip()}\n\nReasoning:\n{reasoning_raw}\n\nANSWER: {final}")

    examples = []
    # GSM8K gold
    for r in gsm:
        reasoning, final = clean_gsm8k(r["answer"])
        if final:
            examples.append({"question": r["question"].strip(),
                             "reasoning_answer": f"{reasoning}\nANSWER: {final}"})
    n_gold = len(examples)

    # RFT self-generated correct solutions
    n_rft = 0
    for line in open("rft_data.jsonl"):
        e = json.loads(line)
        examples.append({"question": e["question"], "reasoning_answer": e["reasoning_answer"]})
        n_rft += 1

    # MetaMath GSM subset for breadth
    mm = load_dataset("meta-math/MetaMathQA", split="train")
    gsm_types = {"GSM_AnsAug", "GSM_Rephrased", "GSM_FOBAR", "GSM_SV"}
    ans_re = re.compile(r"The answer is:\s*(.*?)\s*$", re.DOTALL)
    mm_examples = []
    for r in mm:
        if r["type"] not in gsm_types:
            continue
        m = ans_re.search(r["response"].strip())
        if not m:
            continue
        final = norm_num(m.group(1).split("\n")[0])
        if not re.fullmatch(r"-?\d+(\.\d+)?", final):
            continue
        reasoning = re.sub(r"\\boxed\{([^}]*)\}", r"\1", r["response"][:m.start()].strip())
        if len(reasoning) < 10:
            continue
        mm_examples.append({"question": r["query"].strip(),
                            "reasoning_answer": f"{reasoning}\nANSWER: {final}"})
    random.shuffle(mm_examples)
    mm_examples = mm_examples[:20000]
    examples.extend(mm_examples)

    random.shuffle(examples)
    for e in examples:
        if random.random() < 0.30:
            k = random.randint(3, 8)
            shots = random.sample(fewshot_pool, k)
            shots = [s for s in shots if not s.startswith(e["question"][:60])][:k]
            e["system"] = "\n\n".join(shots)

    with open("train_rft.jsonl", "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print(f"gold={n_gold} rft={n_rft} mm={len(mm_examples)} total={len(examples)} "
          f"fewshot={sum(1 for e in examples if 'system' in e)}")

if __name__ == "__main__":
    main()
