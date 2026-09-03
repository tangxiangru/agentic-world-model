#!/usr/bin/env python3
"""Build the SFT set: prompts rendered exactly as the grader renders them.

Every row is {"prompt": <fully rendered chat string, ends '<start_of_turn>model\\n'>,
              "completion": <chain of thought, 'ANSWER: N', '<end_of_turn>'>}.
The trainer tokenizes both with add_special_tokens=False and masks the prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmt import load_template, user_prompt  # noqa: E402

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
STOP = "<end_of_turn>"
INT_RE = re.compile(r"^-?\d+$")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def clean_gsm8k_cot(answer: str) -> tuple[str, str]:
    """Official GSM8K train answer -> (reasoning without calculator spans, final)."""
    body, _, final = answer.partition("####")
    body = re.sub(r"<<[^>]*>>", "", body)
    return body.strip(), final.strip()


def clean_omi_solution(sol: str, expected: str) -> str | None:
    boxes = BOXED_RE.findall(sol)
    if len(boxes) != 1 or boxes[0].strip() != expected:
        return None
    sol = BOXED_RE.sub(lambda m: m.group(1), sol)
    sol = sol.replace("\\[", "").replace("\\]", "")
    return sol.strip()


def make_row(tok, tpl, question: str, target_body: str, final: str, shots) -> dict | None:
    if "ANSWER:" in target_body:
        return None
    msgs = []
    if shots:
        sysmsg = "\n\n".join(
            f"{q}\n\nReasoning:\n{r}\n\nANSWER: {a}" for q, r, a in shots
        )
        msgs.append({"role": "system", "content": sysmsg})
    msgs.append({"role": "user", "content": user_prompt(question)})
    prompt = tok.apply_chat_template(
        msgs, chat_template=tpl, tokenize=False, add_generation_prompt=True
    )
    completion = f"{target_body}\n\nANSWER: {final}{STOP}"
    return {"prompt": prompt, "completion": completion}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--n-aug", type=int, default=26000)
    ap.add_argument("--per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    tpl = load_template()

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    # exemplar pool for the few-shot-prefixed rows: GSM8K *train* only
    pool = []
    for r in gsm:
        body, final = clean_gsm8k_cot(r["answer"])
        pool.append((r["question"], body, final))

    rows: list[dict] = []
    kept_sources = {"gsm8k_orig": 0, "omi_gsm8k": 0, "omi_aug": 0}

    # 1. official GSM8K train chains
    for q, body, final in pool:
        if not INT_RE.match(final):
            continue
        rows.append(("gsm8k_orig", q, body, final))
        kept_sources["gsm8k_orig"] += 1

    # 2. OpenMathInstruct-2, gsm8k-derived only
    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    omi = omi.filter(
        lambda x: x["problem_source"] in ("gsm8k", "augmented_gsm8k"), num_proc=8
    )
    per_problem: dict[str, int] = {}
    buckets = {"gsm8k": [], "augmented_gsm8k": []}
    for r in omi:
        exp = r["expected_answer"].strip()
        if not INT_RE.match(exp):
            continue
        if len(r["generated_solution"]) > 2200:
            continue
        sol = clean_omi_solution(r["generated_solution"], exp)
        if sol is None:
            continue
        k = r["problem"]
        if per_problem.get(k, 0) >= args.per_problem:
            continue
        per_problem[k] = per_problem.get(k, 0) + 1
        buckets[r["problem_source"]].append((r["problem"], sol, exp))

    for q, sol, exp in buckets["gsm8k"]:
        rows.append(("omi_gsm8k", q, sol, exp))
        kept_sources["omi_gsm8k"] += 1
    aug = buckets["augmented_gsm8k"]
    rng.shuffle(aug)
    for q, sol, exp in aug[: args.n_aug]:
        rows.append(("omi_aug", q, sol, exp))
        kept_sources["omi_aug"] += 1

    rng.shuffle(rows)

    out = []
    n_fs = 0
    for src, q, body, final in rows:
        shots = None
        if rng.random() < args.fewshot_frac:
            k = rng.randint(2, 10)
            shots = rng.sample(pool, k)
            n_fs += 1
        row = make_row(tok, tpl, q, body, final, shots)
        if row is None:
            continue
        row["source"] = src
        out.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    # length report (real tokenization, not a chars/4 guess)
    idx = rng.sample(range(len(out)), min(3000, len(out)))
    lens = []
    for i in idx:
        n = len(tok(out[i]["prompt"], add_special_tokens=False)["input_ids"]) + len(
            tok(out[i]["completion"], add_special_tokens=False)["input_ids"]
        )
        lens.append(n)
    lens.sort()
    report = {
        "n_rows": len(out),
        "sources": kept_sources,
        "n_fewshot_prefixed": n_fs,
        "tok_len_p50": lens[len(lens) // 2],
        "tok_len_p95": lens[int(len(lens) * 0.95)],
        "tok_len_p99": lens[int(len(lens) * 0.99)],
        "tok_len_max": lens[-1],
        "sampled": len(lens),
    }
    with open(args.out.replace(".jsonl", "_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
