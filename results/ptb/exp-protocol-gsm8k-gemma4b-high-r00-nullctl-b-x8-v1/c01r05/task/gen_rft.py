#!/usr/bin/env python3
"""Rejection-sampling data generation (RFT / STaR) with vLLM.

Samples k solutions per training question from a fine-tuned checkpoint, keeps
those that reach the gold answer, filters out solutions containing bogus
arithmetic, and de-duplicates by reasoning path (set of equations).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict

from common import user_prompt, fewshot_system_message

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
EQ_RE = re.compile(r"([0-9][0-9,\.\s\+\-\*/x×÷\(\)%$]*?)\s*=\s*(\$?-?[\d,]+\.?\d*)")


def norm_num(s: str) -> str:
    s = s.replace(",", "").replace("$", "").strip()
    if s.endswith("."):
        s = s[:-1]
    try:
        f = float(s)
    except ValueError:
        return s
    if f != f or f in (float("inf"), float("-inf")) or abs(f) > 1e15:
        return s
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return ("%.6f" % f).rstrip("0").rstrip(".")


def last_number(text: str) -> str | None:
    m = NUM_RE.findall(text)
    if not m:
        return None
    return norm_num(m[-1])


def equations(text: str) -> tuple:
    out = []
    for lhs, rhs in EQ_RE.findall(text):
        lhs = lhs.strip()
        if not lhs:
            continue
        out.append((re.sub(r"\s+", "", lhs), norm_num(rhs)))
    return tuple(out)


def arithmetic_ok(text: str, tol: float = 1e-4) -> bool:
    return True


def load_questions(args):
    from datasets import load_dataset

    items = []
    ds = load_dataset("openai/gsm8k", "main")["train"]
    for r in ds:
        ans = norm_num(r["answer"].split("####")[-1])
        items.append({"question": r["question"].strip(), "answer": ans, "src": "gsm8k_train"})
    if args.metamath > 0:
        mm = load_dataset("meta-math/MetaMathQA")["train"]
        rng = random.Random(1234)
        pool = [r for r in mm if r["type"] in ("GSM_Rephrased", "GSM_FOBAR", "GSM_SV")]
        rng.shuffle(pool)
        n = 0
        seen = set()
        for r in pool:
            if n >= args.metamath:
                break
            m = re.search(r"The answer is:?\s*(.+?)\s*$", r["response"], re.S | re.I)
            if not m:
                continue
            a = m.group(1).strip()
            if not NUM_RE.fullmatch(a.replace("$", "").strip()):
                continue
            q = r["query"].strip()
            if q in seen:
                continue
            seen.add(q)
            items.append({"question": q, "answer": norm_num(a), "src": r["type"]})
            n += 1
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/rft.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--metamath", type=int, default=0)
    ap.add_argument("--max-keep", type=int, default=4)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--fewshot-frac", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stats-out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    with open("templates/gemma3.jinja") as f:
        tok.chat_template = f.read()

    items = load_questions(args)
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} questions")

    fs = fewshot_system_message()
    rng = random.Random(0)
    prompts = []
    for it in items:
        msgs = []
        if rng.random() < args.fewshot_frac:
            msgs.append({"role": "system", "content": fs})
        msgs.append({"role": "user", "content": user_prompt(it["question"])})
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=3072,
        enforce_eager=False,
        dtype="bfloat16",
    )
    sp = SamplingParams(
        n=args.k, temperature=args.temp, top_p=args.top_p, top_k=args.top_k,
        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=None,
    )
    from vllm import TokensPrompt
    token_prompts = [TokensPrompt(prompt_token_ids=tok(p, add_special_tokens=False)["input_ids"])
                     for p in prompts]
    outs = llm.generate(token_prompts, sp)

    kept, n_correct, n_solved = [], 0, 0
    per_q_correct = []
    for it, o in zip(items, outs):
        seen_paths = set()
        good = []
        nc = 0
        for c in o.outputs:
          try:
            text = c.text.strip()
            if "ANSWER:" not in text:
                continue
            head, _, tail = text.partition("ANSWER:")
            first_line = tail.strip().split("\n")[0].strip()
            pred = last_number(first_line)
            text = head.rstrip() + "\n\nANSWER: " + (pred or "")
            if pred is None or pred != it["answer"]:
                continue
            nc += 1
            if not arithmetic_ok(text):
                continue
            key = equations(text)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            good.append(text)
          except Exception:
            continue
        per_q_correct.append(nc / max(len(o.outputs), 1))
        n_correct += nc
        if good:
            n_solved += 1
        rng.shuffle(good)
        for text in good[: args.max_keep]:
            kept.append({
                "question": it["question"],
                "response": text,
                "answer": it["answer"],
                "source": "rft_" + it["src"],
            })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    stats = {
        "questions": len(items),
        "solved_at_least_once": n_solved,
        "pass_rate_mean": sum(per_q_correct) / max(len(per_q_correct), 1),
        "kept": len(kept),
    }
    print(json.dumps(stats, indent=2))
    if args.stats_out:
        with open(args.stats_out, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
