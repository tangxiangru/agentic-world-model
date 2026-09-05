#!/usr/bin/env python3
"""Turn rejection-sampling output into SFT jsonl (same format as prep_data.py)."""
import argparse, json, random, re, collections
from datasets import load_dataset

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ap = argparse.ArgumentParser()
ap.add_argument("--rs", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-per-q", type=int, default=2)
ap.add_argument("--mix", default=None, help="extra sft jsonl to mix in")
ap.add_argument("--n-mix", type=int, default=15000)
ap.add_argument("--fewshot-frac", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=3)
a = ap.parse_args()
rng = random.Random(a.seed)

gsm_train = load_dataset("openai/gsm8k", "main", split="train")
fewshot_pool = []
for r in gsm_train:
    raw = r["answer"].split("####")
    body = raw[0].strip()
    final = raw[-1].strip().replace(",", "")
    fewshot_pool.append(f"{r['question']}\n\nReasoning:\n{body}\n\nANSWER: {final}")

by_q = collections.defaultdict(list)
for path in a.rs:
    for line in open(path):
        r = json.loads(line)
        by_q[r["question"]].append(r["completion"])

records = []
n_q = 0
for q, comps in by_q.items():
    n_q += 1
    rng.shuffle(comps)
    # prefer shorter (more direct) solutions among the correct ones
    comps = comps[: a.max_per_q]
    for c in comps:
        records.append({"prompt": PROMPT_TEMPLATE.format(prompt=q), "completion": c,
                        "src": "rs"})
print(f"RS: {n_q} questions -> {len(records)} samples")

if a.mix:
    extra = [json.loads(l) for l in open(a.mix)]
    rng.shuffle(extra)
    records.extend(extra[: a.n_mix])

rng.shuffle(records)
n_fs = int(len(records) * a.fewshot_frac)
with open(a.out, "w") as f:
    for i, r in enumerate(records):
        p = r["prompt"]
        if i < n_fs and p.startswith("Solve the following math problem"):
            k = rng.randint(1, 10)
            p = "\n\n".join(rng.sample(fewshot_pool, k)) + "\n\n" + p
        f.write(json.dumps({"prompt": p, "completion": r["completion"],
                            "src": r["src"]}) + "\n")
print(f"wrote {len(records)} -> {a.out}")
