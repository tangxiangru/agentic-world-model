#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt -> GSM8K.

Every row is written as {"prompt": <rendered string ending in "<start_of_turn>model\n">,
"completion": <target ending in "<end_of_turn>">, "meta": {...}} so the trainer never
has to re-render anything and preflight can read the target directly.

Sources (all GSM8K *train*-derived or MATH-train-derived; the GSM8K test split is
never touched):
  A  openai/gsm8k train split, minus the 300-item private holdout
  B  nvidia/OpenMathInstruct-2 train_1M, problem_source in {gsm8k, augmented_gsm8k}
  C  nvidia/OpenMathInstruct-2 train_1M, problem_source in {math, augmented_math},
     restricted to numeric answers

Format of the target, matching what the grader reads
(inspect_ai match(numeric=True, location="end") -> last number in the completion):

    <reasoning>

    ANSWER: <number><end_of_turn>
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter

sys.path.insert(0, "/home/ben/task/scripts")
from eval_format import gsm8k_fewshot_system, render, user_prompt  # noqa: E402

CALC = re.compile(r"<<[^>]*>>")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
NUMLIKE = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUMLIKE.match(a.replace(",", "")):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    return a


def clean_gsm8k_reasoning(reasoning: str) -> str:
    return CALC.sub("", reasoning).strip()


def unbox(sol: str) -> str | None:
    """Turn '$\\boxed{18}$' into '18' so the target carries no LaTeX answer marker
    competing with the 'ANSWER: N' line the grader reads."""
    if "\\boxed" not in sol:
        return None
    out = BOXED.sub(r"\1", sol)
    if "\\boxed" in out:
        return None
    # tidy the now-empty math delimiters the box used to fill
    out = re.sub(r"\$\s*(-?[\d,.]+)\s*\$", r"\1", out)
    out = re.sub(r"\\\[\s*(-?[\d,.]+)\s*\\\]", r"\1", out)
    return out.strip()


def make_row(question: str, reasoning: str, answer: str, sysmsg: str | None, src: str) -> dict:
    completion = f"{reasoning.strip()}\n\nANSWER: {answer}<end_of_turn>"
    return {
        "prompt": render(sysmsg, user_prompt(question)),
        "completion": completion,
        "meta": {"source": src, "fewshot": 0 if sysmsg is None else sysmsg.count("Reasoning:")},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/sft_v1.jsonl")
    ap.add_argument("--n-omi-gsm8k", type=int, default=60000)
    ap.add_argument("--n-omi-math", type=int, default=10000)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    from datasets import load_dataset

    holdout = set(json.load(open("/home/ben/task/data/train_holdout_ids.json")))
    rows: list[dict] = []

    # ---- A: gsm8k train (minus holdout) ------------------------------------
    tr = load_dataset("openai/gsm8k", "main", split="train")
    gsm_pool = []
    for i, rec in enumerate(tr):
        if i in holdout:
            continue
        parts = rec["answer"].split("####")
        ans = norm_answer(parts[-1])
        if ans is None:
            continue
        gsm_pool.append((rec["question"], clean_gsm8k_reasoning("####".join(parts[:-1])), ans))
    print(f"A gsm8k train usable: {len(gsm_pool)}", flush=True)

    # ---- B/C: OpenMathInstruct-2 -------------------------------------------
    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    gsm_src = {"gsm8k", "augmented_gsm8k"}
    math_src = {"math", "augmented_math"}
    b_pool, c_pool = [], []
    seen_pairs = set()
    for rec in omi:
        ps = rec["problem_source"]
        if ps not in gsm_src and ps not in math_src:
            continue
        ans = norm_answer(rec["expected_answer"])
        if ans is None:
            continue
        sol = unbox(rec["generated_solution"])
        if sol is None or len(sol) < 20:
            continue
        key = (hash(rec["problem"]) & 0xFFFFFFFF, ans)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        (b_pool if ps in gsm_src else c_pool).append((rec["problem"], sol, ans))
    print(f"B omi-gsm8k usable: {len(b_pool)}  C omi-math usable: {len(c_pool)}", flush=True)

    rng.shuffle(b_pool)
    rng.shuffle(c_pool)
    b_pool = b_pool[: args.n_omi_gsm8k]
    c_pool = c_pool[: args.n_omi_math]

    items: list[tuple[str, str, str, str]] = []
    for _ in range(args.gsm8k_repeat):
        items += [(q, r, a, "gsm8k_train") for q, r, a in gsm_pool]
    items += [(q, r, a, "omi_gsm8k") for q, r, a in b_pool]
    items += [(q, r, a, "omi_math") for q, r, a in c_pool]
    rng.shuffle(items)

    # ---- few-shot prefixes on a small slice --------------------------------
    # The grader always prepends a fixed 10-shot system message (2043 tokens).
    # Teach the model to ignore the demos and stop after the final answer, without
    # paying the 2k-token prompt on every row.
    demo_pool = [(q, r, a) for q, r, a in gsm_pool]
    rng.shuffle(demo_pool)
    demo_pool = demo_pool[:400]

    def fewshot_sys(k: int) -> str:
        picks = rng.sample(demo_pool, k)
        return "\n\n".join(
            f"{q}\n\nReasoning:\n{r}\n\nANSWER: {a}" for q, r, a in picks
        )

    n_fs = int(len(items) * args.fewshot_frac)
    for i, (q, r, a, src) in enumerate(items):
        if i < n_fs:
            k = rng.choice([1, 2, 3, 5, 8, 10])
            rows.append(make_row(q, r, a, fewshot_sys(k), src))
        else:
            rows.append(make_row(q, r, a, None, src))
    rng.shuffle(rows)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", args.out, len(rows))
    print(Counter(r["meta"]["source"] for r in rows))
    print("fewshot rows:", sum(1 for r in rows if r["meta"]["fewshot"]))

    # a plain-text dump for the contamination checker (question + target text)
    dump = args.out.replace(".jsonl", "_decon.jsonl")
    with open(dump, "w") as f:
        for q, r, a, _ in items:
            f.write(json.dumps({"question": q, "answer": f"{r}\n\nANSWER: {a}"}) + "\n")
    print("wrote", dump)


if __name__ == "__main__":
    main()
