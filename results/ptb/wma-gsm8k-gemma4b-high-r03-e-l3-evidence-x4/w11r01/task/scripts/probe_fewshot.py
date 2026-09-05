#!/usr/bin/env python3
"""Does the grader's 10-shot system prefix help or hurt the tuned model?

The harness always prepends it; exp-02 trained only 10% of rows with a k-shot
prefix. Measured on GSM8K TRAIN questions (never the test split) so this is a
probe, not a benchmark read.
"""
import json, random, re, sys, argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
sys.path.insert(0, "scripts")
from train_sft import PROMPT_TEMPLATE, SNAPSHOT, TEMPLATE, eval_fewshots

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=800)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main")["train"]
rng = random.Random(123)
idx = rng.sample(range(len(ds)), a.n)
qs = [(ds[i]["question"], ds[i]["answer"].split("####")[-1].strip().replace(",", "")) for i in idx]

tok = AutoTokenizer.from_pretrained(SNAPSHOT)
tmpl = open(TEMPLATE).read()
shots = "\n\n".join(eval_fewshots())

def render(q, with_shots):
    m = ([{"role": "system", "content": shots}] if with_shots else []) + \
        [{"role": "user", "content": PROMPT_TEMPLATE.format(prompt=q)}]
    return tok.apply_chat_template(m, chat_template=tmpl, tokenize=False, add_generation_prompt=True)

llm = LLM(model=a.model, gpu_memory_utilization=0.85, max_model_len=4096, dtype="bfloat16", seed=0)
sp = SamplingParams(n=1, temperature=0.0, max_tokens=768, stop_token_ids=[1, 106])
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

def score(prompts):
    outs = llm.generate(prompts, sp)
    ok = 0
    for o, (_, gold) in zip(outs, qs):
        t = o.outputs[0].text
        m = NUM.findall(t)
        if m and m[-1].replace(",", "").rstrip(".") == gold:
            ok += 1
    return ok / len(qs)

res = {"n": a.n,
       "acc_10shot": score([render(q, True) for q, _ in qs]),
       "acc_0shot": score([render(q, False) for q, _ in qs])}
res["delta_10shot_minus_0shot"] = res["acc_10shot"] - res["acc_0shot"]
print(json.dumps(res, indent=2))
json.dump(res, open(a.out, "w"), indent=2)
