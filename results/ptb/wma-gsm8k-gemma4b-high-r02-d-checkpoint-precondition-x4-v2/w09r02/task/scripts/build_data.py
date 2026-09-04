#!/usr/bin/env python3
"""Build GSM8K-style SFT data rendered for the grading template.

Sources (all GSM8K *train* derived or independent; never the test split):
  - nvidia/OpenMathInstruct-2, problem_source in {gsm8k, augmented_gsm8k}
  - openai/gsm8k train split (native reasoning, calculator annotations stripped)

Target shape: chain of thought, then a final line 'ANSWER: <integer>'.
The row is stored as {prompt, completion} where completion already ends with
the grading template's terminator '<end_of_turn>'.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

OMI2_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"
SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"
STOP = "<end_of_turn>"

BOXED = re.compile(r"\\boxed\{")
CALC = re.compile(r"<<[^>]*>>")


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{X} with X (brace-balanced)."""
    while True:
        m = BOXED.search(text)
        if m is None:
            return text
        start = m.end()  # just after '{'
        depth = 1
        i = start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth:
            return text[: m.start()] + text[start:]
        text = text[: m.start()] + text[start : i - 1] + text[i:]


def is_int_answer(a: str) -> bool:
    a = a.strip().replace(",", "").replace("$", "")
    if a.startswith("-"):
        a = a[1:]
    if a.endswith(".0"):
        a = a[:-2]
    return a.isdigit() and len(a) <= 12


def norm_int(a: str) -> str:
    a = a.strip().replace(",", "").replace("$", "")
    if a.endswith(".0"):
        a = a[:-2]
    return str(int(a))


def qkey(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def load_omi2(max_per_question: int, rng: random.Random):
    files = sorted(glob.glob(OMI2_GLOB))
    assert files, "OpenMathInstruct-2 parquet shards not found"
    by_q = defaultdict(list)
    for f in files:
        t = pq.read_table(f, columns=["problem", "generated_solution", "expected_answer", "problem_source"])
        srcs = t.column("problem_source").to_pylist()
        probs = t.column("problem").to_pylist()
        sols = t.column("generated_solution").to_pylist()
        ans = t.column("expected_answer").to_pylist()
        for s, p, sol, a in zip(srcs, probs, sols, ans):
            if s not in ("gsm8k", "augmented_gsm8k"):
                continue
            if not is_int_answer(a):
                continue
            by_q[qkey(p)].append((p, sol, norm_int(a)))
    rows = []
    for k, cands in by_q.items():
        rng.shuffle(cands)
        seen_sol = set()
        kept = 0
        for p, sol, a in cands:
            body = strip_boxed(sol).strip()
            if "\\boxed" in body:
                continue
            # solution must not be dominated by latex display math walls
            if len(body) < 30 or len(body) > 4000:
                continue
            h = hash(re.sub(r"\s+", " ", body))
            if h in seen_sol:
                continue
            seen_sol.add(h)
            rows.append({"question": p.strip(), "body": body, "answer": a, "src": "omi2"})
            kept += 1
            if kept >= max_per_question:
                break
    return rows


def load_gsm8k_train():
    from datasets import load_dataset

    d = load_dataset("openai/gsm8k", "main")["train"]
    rows = []
    for r in d:
        ans = r["answer"]
        body, _, final = ans.rpartition("####")
        body = CALC.sub("", body).strip()
        final = final.strip().replace(",", "")
        if not is_int_answer(final):
            continue
        rows.append({"question": r["question"].strip(), "body": body, "answer": norm_int(final), "src": "gsm8k_train"})
    return rows


def make_fewshot_block(pool, k, rng):
    picks = rng.sample(pool, k)
    parts = []
    for p in picks:
        parts.append(f"{p['question']}\n\nReasoning:\n{p['body']}\n\nANSWER: {p['answer']}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-omi2", type=int, default=60000)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--gsm8k-native-repeat", type=int, default=1)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--fewshot-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    template = open(TEMPLATE_PATH).read()

    native = load_gsm8k_train()
    print(f"gsm8k train native rows: {len(native)}")
    omi2 = load_omi2(args.max_per_question, rng)
    print(f"omi2 gsm8k-style rows: {len(omi2)}")
    rng.shuffle(omi2)
    omi2 = omi2[: args.n_omi2]

    rows = omi2 + native * args.gsm8k_native_repeat
    rng.shuffle(rows)

    n_fs = int(len(rows) * args.fewshot_frac)
    n_written = 0
    n_skipped = 0
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            body = r["body"]
            # one answer marker only (pitfalls.yaml double_answer_format)
            if "ANSWER:" in body or "####" in body:
                n_skipped += 1
                continue
            user = MATH_PROMPT_TEMPLATE.format(prompt=r["question"])
            msgs = []
            if i < n_fs:
                msgs.append({"role": "system", "content": make_fewshot_block(native, args.fewshot_k, rng)})
            msgs.append({"role": "user", "content": user})
            prompt = tok.apply_chat_template(
                msgs, chat_template=template, tokenize=False, add_generation_prompt=True
            )
            completion = f"{body}\n\nANSWER: {r['answer']}{STOP}"
            f.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "answer": r["answer"],
                        "src": r["src"],
                        "fewshot": i < n_fs,
                    }
                )
                + "\n"
            )
            n_written += 1
    print(
        f"wrote {n_written} rows to {args.out} "
        f"({n_fs} tagged for a {args.fewshot_k}-shot prefix, {n_skipped} skipped for a duplicate answer marker)"
    )


if __name__ == "__main__":
    main()
