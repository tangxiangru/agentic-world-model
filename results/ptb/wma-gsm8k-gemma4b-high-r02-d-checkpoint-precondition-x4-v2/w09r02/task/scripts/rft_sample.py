#!/usr/bin/env python3
"""Rejection-sampling data: sample k solutions per GSM8K *train* question from a
checkpoint, keep the ones whose final 'ANSWER: <n>' equals the gold answer, and
write them in the same pre-rendered {prompt, completion} shape as build_data.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"
STOP = "<end_of_turn>"
ANS = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")


def final_answer(text: str):
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return None
    m = ANS.match(lines[-1].strip())
    if not m:
        return None
    v = m.group(1).replace(",", "")
    try:
        f = float(v)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-keep", type=int, default=2)
    ap.add_argument("--adaptive-keep", action="store_true",
                    help="keep fewer samples for problems the model already solves every time, "
                         "more for the ones it only occasionally gets right")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    template = open(TEMPLATE_PATH).read()

    d = load_dataset("openai/gsm8k", "main")["train"]
    items = []
    for r in d:
        gold = r["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
        try:
            gold = str(int(float(gold)))
        except ValueError:
            continue
        items.append((r["question"].strip(), gold))
    if args.limit:
        items = items[: args.limit]
    print(f"[rft] {len(items)} train questions", flush=True)

    prompts = []
    for q, _ in items:
        msgs = [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}]
        prompts.append(tok.apply_chat_template(msgs, chat_template=template, tokenize=False, add_generation_prompt=True))

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=1536,
              dtype="bfloat16", enforce_eager=False, seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    n_kept = 0
    n_solved = 0
    per_q_correct = []
    with open(args.out, "w") as f:
        for (q, gold), o, pr in zip(items, outs, prompts):
            cands = []
            n_c = 0
            for c in o.outputs:
                txt = c.text.strip()
                if final_answer(txt) == gold:
                    n_c += 1
                    body = re.sub(r"\s+", " ", txt)
                    cands.append((txt, body))
            per_q_correct.append(n_c / max(1, len(o.outputs)))
            n_solved += n_c > 0
            seen = set()
            uniq = []
            for txt, body in cands:
                if body in seen:
                    continue
                seen.add(body)
                uniq.append(txt)
            rng.shuffle(uniq)
            keep = args.max_keep
            if args.adaptive_keep and cands:
                frac = n_c / max(1, len(o.outputs))
                keep = 1 if frac >= 0.875 else (2 if frac >= 0.5 else args.max_keep)
            for txt in uniq[:keep]:
                if txt.count("ANSWER:") != 1:
                    continue
                f.write(json.dumps({"prompt": pr, "completion": txt + STOP, "answer": gold, "src": "rft", "fewshot": False}) + "\n")
                n_kept += 1
    frac = sum(per_q_correct) / len(per_q_correct)
    stats = {"questions": len(items), "solved_at_least_once": n_solved,
             "pass_at_%d" % args.k: n_solved / len(items), "mean_sample_accuracy": frac, "rows_kept": n_kept}
    print("[rft]", json.dumps(stats), flush=True)
    if args.stats_out:
        json.dump(stats, open(args.stats_out, "w"), indent=1)


if __name__ == "__main__":
    main()
