#!/usr/bin/env python3
"""Diagnostic: how much does the harness's 10-shot system prefix change accuracy?

Replicates the inspect_evals/gsm8k prompt exactly (same MATH_PROMPT_TEMPLATE, same
few-shot construction and seed, same templates/gemma3.jinja, same last-number
grading rule) and runs it at several shot counts inside ONE vllm process, so the
arms share a scheduler and the run-to-run drift measured in exp-08 is common-mode.

Reads the benchmark's test split through the harness's own loader for the prompts
only; no test text is written anywhere.
"""
import argparse, json, sys, os

from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import (MATH_PROMPT_TEMPLATE, record_to_sample,
                                       sample_to_fewshot)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_rft import final_answer, norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--shots", default="10,2,0")
ap.add_argument("--limit", type=int, default=1319)
ap.add_argument("--out", required=True)
ap.add_argument("--template", default="templates/gemma3.jinja")
a = ap.parse_args()

fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                      sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
blocks = [sample_to_fewshot(s) for s in fewshots]
test = hf_dataset(path="openai/gsm8k", data_dir="main", split="test",
                  sample_fields=record_to_sample, limit=a.limit)

template = open(a.template).read()
tok = AutoTokenizer.from_pretrained(a.model)
eot = tok.convert_tokens_to_ids("<end_of_turn>")

arms = [int(x) for x in a.shots.split(",")]
prompts, tags = [], []
for k in arms:
    sysmsg = "\n\n".join(blocks[:k]) if k else None
    for s in test:
        msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + \
               [{"role": "user", "content": MATH_PROMPT_TEMPLATE.replace("{prompt}", s.input)}]
        prompts.append(tok.apply_chat_template(msgs, chat_template=template,
                                               tokenize=False, add_generation_prompt=True))
        tags.append((k, s.target))

llm = LLM(model=a.model, gpu_memory_utilization=0.85, max_model_len=4096,
          dtype="bfloat16", max_num_seqs=512, enable_prefix_caching=True)
outs = llm.generate(prompts, SamplingParams(n=1, temperature=0.0, max_tokens=1024,
                                            stop_token_ids=[eot]))

res = {k: [0, 0] for k in arms}
for (k, gold), o in zip(tags, outs):
    t = o.outputs[0].text
    res[k][1] += 1
    if final_answer(t) is not None and final_answer(t) == norm(gold):
        res[k][0] += 1
summary = {str(k): {"correct": v[0], "n": v[1], "accuracy": v[0] / v[1]} for k, v in res.items()}
print(json.dumps(summary, indent=2))
json.dump(summary, open(a.out, "w"), indent=2)
