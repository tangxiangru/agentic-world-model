#!/usr/bin/env python3
"""Score a checkpoint on data/heldout_dev300.jsonl - 300 openai/gsm8k *train*
rows that were excluded from every training corpus in this session.

Uses the grader's prompt verbatim (10-shot system message from
data/fewshot_system.txt + MATH_PROMPT_TEMPLATE, rendered through
templates/gemma3.jinja) and the grader's scoring rule
(match(numeric=True, location="end") = last number in the output), so the
number is comparable in kind to evaluate.py, on 2x the items and without
touching the benchmark test set.
"""
import argparse, json, re, sys
sys.path.insert(0, ".")
from build_data import render_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--gpu-mem", type=float, default=0.85)
ap.add_argument("--nshot", type=int, default=10,
                help="10 = the grader's exact 10-shot prefix; 0 = no prefix")
a = ap.parse_args()

rows = [json.loads(l) for l in open("data/heldout_dev300.jsonl")]
sysmsg = open("data/fewshot_system.txt").read() if a.nshot else None
prompts = ["<bos>" + render_prompt(r["question"], sysmsg) for r in rows]

from vllm import LLM, SamplingParams
llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=8192,
          dtype="bfloat16", enforce_eager=False)
sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens,
                    stop_token_ids=[1, 106])
outs = llm.generate(prompts, sp)

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def graded(text):
    """match(numeric=True, location='end'): last numeric token wins."""
    m = NUM.findall(text.replace(",", "").replace("$", ""))
    return m[-1] if m else None


res, ok, trunc = [], 0, 0
for r, o in zip(rows, outs):
    t = o.outputs[0].text
    g = graded(t)
    gold = r["gold"].replace(",", "").replace("$", "")
    hit = g is not None and abs(float(g) - float(gold)) < 1e-6
    ok += hit
    trunc += o.outputs[0].finish_reason == "length"
    res.append({"id": r["id"], "gold": r["gold"], "pred": g, "correct": bool(hit),
                "n_tok": len(o.outputs[0].token_ids), "text": t})
summary = {"model": a.model, "n": len(rows), "accuracy": round(ok / len(rows), 4),
           "truncated": trunc}
print(json.dumps(summary, indent=2))
json.dump({"summary": summary, "samples": res}, open(a.out, "w"), indent=1)
