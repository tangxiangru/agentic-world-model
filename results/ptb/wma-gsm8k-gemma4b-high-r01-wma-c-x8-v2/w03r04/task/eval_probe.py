#!/usr/bin/env python3
"""Fast offline vLLM scorer on the held-out probe set, replicating the grader.

Same prompt (10-shot system block built exactly like inspect_evals.gsm8k, then
MATH_PROMPT_TEMPLATE), same chat template, same scoring rule (last numeric
token of the completion, compared numerically). Used as a secondary signal for
checkpoint selection; the graded protocol stays evaluate.py --limit 150.
"""
from __future__ import annotations

import argparse
import json
import re

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from train_sft import BASE, MATH_PROMPT_TEMPLATE, fewshot_block

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str):
    m = NUM_RE.findall(text.replace("$", "").replace(",", ""))
    if not m:
        return None
    try:
        return float(m[-1].rstrip("."))
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", default="data/probe200.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fewshot", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.probe)]
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = open("templates/gemma3.jinja").read()

    # the grader's own few-shot block: first 10 of gsm8k train shuffled with seed 42
    shots = []
    if args.fewshot:
        g = load_dataset("openai/gsm8k", "main")["train"].shuffle(seed=42).select(range(args.fewshot))
        for r in g:
            body, ans = r["answer"].rsplit("####", 1)
            shots.append((r["question"], body.strip(), ans.strip()))

    prompts = []
    for r in rows:
        msgs = []
        if shots:
            msgs.append({"role": "system", "content": fewshot_block(shots)})
        msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=r["question"])})
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    llm = LLM(model=args.model, tokenizer=args.tokenizer or args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=4096, seed=0)
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
    )
    outs = llm.generate(prompts, sp)

    n_ok = n_stop = 0
    recs = []
    for r, o in zip(rows, outs):
        t = o.outputs[0].text
        v = last_number(t)
        try:
            gold = float(r["gold"].replace(",", ""))
        except ValueError:
            continue
        ok = v is not None and abs(v - gold) < 1e-6
        n_ok += ok
        n_stop += o.outputs[0].finish_reason == "stop"
        recs.append({"id": r["id"], "gold": r["gold"], "read": v, "correct": bool(ok), "tail": t[-200:]})
    res = {
        "model": args.model,
        "n": len(recs),
        "accuracy": n_ok / max(1, len(recs)),
        "clean_stop_share": n_stop / max(1, len(recs)),
        "fewshot": args.fewshot,
        "temperature": args.temperature,
    }
    print(json.dumps(res, indent=1))
    res["items"] = recs
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
