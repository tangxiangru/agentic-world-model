import json, random
random.seed(42)

def load(path):
    return [json.loads(l) for l in open(path)]

gsm = load("data/gsm8k_sft.jsonl")
meta = load("data/metamath_gsm_sft.jsonl")
random.shuffle(meta)
meta_sample = meta[:30000]

# upweight the gold gsm8k train 2x
combined = gsm*2 + meta_sample
random.shuffle(combined)

with open("data/combined3_sft.jsonl", "w") as f:
    for ex in combined:
        f.write(json.dumps(ex) + "\n")
print("gsm", len(gsm), "meta_sample", len(meta_sample), "combined", len(combined))
