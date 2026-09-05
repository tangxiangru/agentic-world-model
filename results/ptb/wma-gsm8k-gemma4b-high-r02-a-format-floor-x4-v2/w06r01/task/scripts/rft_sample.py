#!/usr/bin/env python3
"""Rejection-sampling data: draw k completions per problem from a trained
checkpoint, keep the ones whose final number matches the gold answer.

Prompts are rendered with templates/gemma3.jinja - the grader's own template -
so the samples are exactly on the distribution the model is graded on.  The
gold answers come from GSM8K TRAIN / OpenMathInstruct-2 problems only.
"""
import argparse
import json
import random
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

TASK = "/home/ben/task"
SNAPSHOT = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
            "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(txt: str):
    m = NUM.findall(txt.replace("$", ""))
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", default=f"{TASK}/data/pool2.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-problems", type=int, default=32000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--gpu-util", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.pool)]
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n_problems]

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    tok.chat_template = open(f"{TASK}/templates/gemma3.jinja").read()
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": r["prompt"]}],
        tokenize=False, add_generation_prompt=True) for r in rows]

    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util,
              max_model_len=4096, dtype="bfloat16", seed=a.seed)
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p,
                        max_tokens=a.max_tokens, stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    kept = n_solved = 0
    with open(a.out, "w") as fh:
        for r, o in zip(rows, outs):
            try:
                gold = float(r["answer"])
            except ValueError:
                continue
            good, seen = [], set()
            for c in o.outputs:
                t = c.text.strip()
                if "ANSWER:" not in t:
                    continue
                v = last_number(t.rsplit("ANSWER:", 1)[1])
                if v is None or abs(v - gold) > 1e-6:
                    continue
                key = re.sub(r"\s+", " ", t)
                if key in seen:
                    continue
                seen.add(key)
                good.append(t)
            if good:
                n_solved += 1
            good.sort(key=len)
            for t in good[: a.keep_per_problem]:
                fh.write(json.dumps({
                    "question": r["question"], "prompt": r["prompt"],
                    "target": t + "<end_of_turn>", "answer": r["answer"],
                    "src": "rft:self"}) + "\n")
                kept += 1
    print(f"problems={len(rows)} solved_at_least_once={n_solved} "
          f"({n_solved/len(rows):.3f}) rows_kept={kept}")


if __name__ == "__main__":
    main()
