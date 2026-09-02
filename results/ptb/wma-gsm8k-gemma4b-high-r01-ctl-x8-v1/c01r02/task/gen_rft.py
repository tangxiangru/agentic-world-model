#!/usr/bin/env python3
"""Rejection-sampling fine-tuning data: sample k solutions per GSM8K-TRAIN question
from the current checkpoint, keep the ones whose graded answer is right.

The samples are drawn through the grader's own prompt render and graded with the
grader's own rule (last numeric word of the completion, inspect_ai match(numeric=True,
location="end")), so a kept row is a row the harness would have scored correct.
Only questions from the gsm8k TRAIN split are used; the test split is never touched.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()
STOP = "<end_of_turn>"


def norm(x: str) -> str:
    x = x.strip().replace(",", "").replace("$", "")
    if x.endswith(".0"):
        x = x[:-2]
    return x


def graded_answer(text: str) -> str | None:
    """Reproduce inspect_ai match(numeric=True, location='end'): last numeric word."""
    v = text.strip().replace(",", "").replace("$", "")
    words = re.split(r"\s+", v)
    words.reverse()
    for w in words:
        w2 = w.strip(".:;!?()[]{}\"'")
        if w2.replace(".", "").replace("-", "").isnumeric():
            return norm(w2)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--varshot-frac", type=float, default=0.08)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--stats", default=None)
    a = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = open("/home/ben/task/templates/gemma3.jinja").read()
    eval_sys = open("/home/ben/task/data/eval_system_message.txt").read()

    ds = load_from_disk("/home/ben/task/data/gsm8k_raw")["train"]
    qs = []
    for q, ans in zip(ds["question"], ds["answer"]):
        gold = norm(ans.split("####")[-1].strip())
        qs.append((q.strip(), gold))
    if a.limit:
        qs = qs[: a.limit]

    demo_pool = None

    def render(question: str) -> str:
        nonlocal demo_pool
        u = rng.random()
        msgs = []
        if u < a.fewshot_frac:
            msgs.append({"role": "system", "content": eval_sys})
        elif u < a.fewshot_frac + a.varshot_frac:
            if demo_pool is None:
                demo_pool = []
                for q, ans in list(zip(ds["question"], ds["answer"]))[:400]:
                    body, _, g = ans.partition("####")
                    body = re.sub(r"<<[^>]*>>", "", body).strip()
                    demo_pool.append(f"{q.strip()}\n\nReasoning:\n{body}\n\nANSWER: {norm(g.strip())}")
            k = rng.choice([2, 4, 6, 8])
            msgs.append({"role": "system", "content": "\n\n".join(rng.sample(demo_pool, k))})
        msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=question)})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    prompts = [render(q) for q, _ in qs]

    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=4096,
              dtype="bfloat16", enforce_eager=False, seed=a.seed)
    # no per-request seed: with n>1 a fixed seed risks collapsing the k samples
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p,
                        max_tokens=a.max_tokens)
    outs = llm.generate(prompts, sp)

    n_kept, n_solved, rows = 0, 0, []
    per_q = []
    for (q, gold), prompt, o in zip(qs, prompts, outs):
        good = []
        for c in o.outputs:
            txt = c.text
            if graded_answer(txt) != gold:
                continue
            txt = txt.strip()
            # the completion must be self-contained and correctly terminated
            if "ANSWER:" not in txt:
                continue
            txt = txt[: txt.rindex("ANSWER:")] .rstrip() + f"\n\nANSWER: {gold}"
            if len(txt) < 30 or len(txt) > 4000:
                continue
            good.append(txt)
        per_q.append(len(good))
        if not good:
            continue
        n_solved += 1
        # prefer the shortest distinct solutions: shorter correct chains are cleaner
        uniq, seen = [], set()
        for t in sorted(good, key=len):
            key = re.sub(r"\s+", " ", t)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(t)
        # up-weight the questions the model finds hard: keep more distinct solutions
        # where the pass rate is low, fewer where it already solves the item every time
        keep_n = a.keep_per_question if len(good) / a.k >= 0.75 else a.keep_per_question * 2
        for t in uniq[:keep_n]:
            rows.append({"prompt": prompt, "completion": t + STOP, "target": t + STOP,
                         "answer": gold, "src": "rft_self", "question": q})
            n_kept += 1

    rng.shuffle(rows)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    stats = {
        "questions": len(qs), "k": a.k, "kept_rows": n_kept,
        "questions_with_at_least_one_correct": n_solved,
        "pass_at_k": round(n_solved / max(1, len(qs)), 4),
        "mean_correct_per_question": round(sum(per_q) / max(1, len(per_q)), 3),
        "unsolved_questions": len(qs) - n_solved,
    }
    print(json.dumps(stats, indent=2))
    if a.stats:
        json.dump(stats, open(a.stats, "w"), indent=2)


if __name__ == "__main__":
    main()
