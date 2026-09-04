#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt -> GSM8K.

Every row is rendered with the *same* chat template the grader uses
(templates/gemma3.jinja, hash-checked) and every target ends with the same
terminator the grader stops on (<end_of_turn>), so training and grading render
the identical string for the identical conversation.

Sources (all GSM8K *train*-derived or MATH-train-derived; the GSM8K test split
is never read here):
  * openai/gsm8k train  - gold human chains, calculator annotations stripped
  * nvidia/OpenMathInstruct-2 - Llama-3.1-405B solutions to GSM8K-train and
    MATH-train problems and to augmentations of them

Output: jsonl with {prompt, completion, text, source, n_tokens}.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

TEMPLATE_PATH = "templates/gemma3.jinja"
BASE = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")

# byte-for-byte copy of inspect_evals.gsm8k.MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOXED = re.compile(r"\\boxed\s*\{")
CALC = re.compile(r"<<[^>]*>>")
INT_RE = re.compile(r"^-?\d+$")


def strip_boxed(s: str) -> str:
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
    while True:
        m = BOXED.search(s)
        if not m:
            return s
        i = m.end()  # just after '{'
        depth = 1
        while i < len(s) and depth:
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
            i += 1
        if depth:
            return s[: m.start()] + s[m.end():]
        s = s[: m.start()] + s[m.end(): i - 1] + s[i:]


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").replace("\\!", "")
    a = a.replace("\\%", "").replace("%", "").strip()
    if a.endswith(".0"):
        a = a[:-2]
    if INT_RE.match(a):
        return str(int(a))
    return None


def fewshot_prefix() -> str:
    """Reproduce the harness's 10-shot system message exactly."""
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

    shots = hf_dataset(
        path="openai/gsm8k", data_dir="main", split="train",
        sample_fields=record_to_sample, shuffle=True, seed=42, limit=10,
    )
    return "\n\n".join(sample_to_fewshot(s) for s in shots)


def gsm8k_gold():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")["train"]
    for r in ds:
        body, _, ans = r["answer"].partition("####")
        ans = norm_answer(ans)
        if ans is None:
            continue
        body = CALC.sub("", body).strip()
        yield r["question"].strip(), f"{body}\n\nANSWER: {ans}", ans, "gsm8k_train_gold"


def openmath(shards, keep_sources, max_per_problem):
    import pyarrow.parquet as pq
    per = defaultdict(int)
    for path in shards:
        t = pq.read_table(path, columns=["problem", "generated_solution",
                                         "expected_answer", "problem_source"])
        for r in t.to_pylist():
            src = r["problem_source"]
            if src not in keep_sources:
                continue
            ans = norm_answer(r["expected_answer"] or "")
            if ans is None:
                continue
            q = (r["problem"] or "").strip()
            if per[q] >= max_per_problem:
                continue
            sol = strip_boxed(r["generated_solution"] or "").strip()
            if not sol or "####" in sol:
                continue
            # the trailing number of the body must not fight the ANSWER line
            sol = re.sub(r"\s+", " ", sol) if False else sol
            per[q] += 1
            yield q, f"{sol}\n\nANSWER: {ans}", ans, src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--max-gsm-aug", type=int, default=60000)
    ap.add_argument("--max-math", type=int, default=12000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    template = open(TEMPLATE_PATH).read()
    print("template sha256", hashlib.sha256(template.encode()).hexdigest())

    rng = random.Random(args.seed)
    prefix = fewshot_prefix()
    print("fewshot prefix chars", len(prefix), "tokens", len(tok(prefix).input_ids))

    shards = sorted(glob.glob(os.path.expanduser(
        "~/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet")))
    print(f"{len(shards)} OpenMathInstruct-2 shards")

    rows, seen = [], set()

    def add(q, comp, ans, src):
        key = hashlib.md5((q + "||" + comp).encode()).hexdigest()
        if key in seen:
            return False
        seen.add(key)
        rows.append({"question": q, "completion": comp, "answer": ans, "source": src})
        return True

    for q, c, a, s in gsm8k_gold():
        add(q, c, a, s)
    n_gold = len(rows)
    print("gsm8k gold", n_gold)

    n = 0
    for q, c, a, s in openmath(shards, {"gsm8k", "augmented_gsm8k"},
                               args.max_per_problem):
        if add(q, c, a, s):
            n += 1
        if n >= args.max_gsm_aug:
            break
    print("openmath gsm8k-family", n)

    m = 0
    for q, c, a, s in openmath(shards, {"math", "augmented_math"}, 1):
        if add(q, c, a, s):
            m += 1
        if m >= args.max_math:
            break
    print("openmath math-family", m)

    # ---- render ------------------------------------------------------------
    rng.shuffle(rows)
    out, dropped = [], 0
    for i, r in enumerate(rows):
        user = MATH_PROMPT_TEMPLATE.format(prompt=r["question"])
        msgs = [{"role": "user", "content": user}]
        if rng.random() < args.fewshot_frac:
            msgs = [{"role": "system", "content": prefix}] + msgs
        prompt = tok.apply_chat_template(msgs, chat_template=template,
                                         tokenize=False, add_generation_prompt=True)
        completion = r["completion"] + "<end_of_turn>\n"
        n_tok = len(tok(prompt + completion).input_ids)
        if n_tok > args.max_tokens:
            dropped += 1
            continue
        out.append({"prompt": prompt, "completion": completion,
                    "text": prompt + completion, "source": r["source"],
                    "answer": r["answer"], "question": r["question"],
                    "n_tokens": n_tok})
    print(f"rendered {len(out)}; dropped {dropped} over {args.max_tokens} tokens")
    print("sources", Counter(o["source"] for o in out))
    lens = sorted(o["n_tokens"] for o in out)
    print(f"tokens p50={lens[len(lens)//2]} p95={lens[int(len(lens)*.95)]} max={lens[-1]}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print("wrote", args.out)

    # a plain-text copy for the contamination checker
    with open(args.out.replace(".jsonl", "_check.jsonl"), "w") as f:
        for o in out:
            f.write(json.dumps({"text": o["question"] + "\n" + o["completion"]}) + "\n")
    print("wrote", args.out.replace(".jsonl", "_check.jsonl"))


if __name__ == "__main__":
    main()
