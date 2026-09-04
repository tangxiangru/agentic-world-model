#!/usr/bin/env python3
"""Build the SFT pool for GSM8K from OpenMathInstruct-2 (GSM8K-train-derived only)
plus the openai/gsm8k *train* split gold solutions.

Output jsonl rows: {"prompt": <rendered up to <start_of_turn>model\n>,
                    "completion": <body + "<end_of_turn>">,
                    "question": ..., "answer": ..., "src": ...}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402

OMI2_DIR = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/469216e3f46f4dacf476b382e192485ea51a143e/data"
GSM8K_TRAIN = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet"

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
CALC = re.compile(r"<<[^>]*>>")
NUM_OK = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def unbox(text: str) -> str:
    """Turn \\boxed{x} into x, and drop the surrounding \\[ \\] display math if it
    became a bare number. Keeps the prose intact."""
    prev = None
    while prev != text:
        prev = text
        text = BOXED.sub(r"\1", text)
    return text


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUM_OK.match(a):
        return None
    if "." in a:
        f = float(a)
        return str(int(f)) if f == int(f) else ("%g" % f)
    return str(int(a))


def load_omi2(splits):
    rows = []
    paths = [q for sp in splits for q in sorted(glob.glob(f"{OMI2_DIR}/{sp}-*.parquet"))]
    print("omi2 files:", len(paths), flush=True)
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=50000,
                                     columns=["problem", "generated_solution",
                                              "expected_answer", "problem_source"]):
            for r in batch.to_pylist():
                if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                    continue
                rows.append(r)
    return rows


def load_gsm8k_train():
    out = []
    for r in pq.ParquetFile(GSM8K_TRAIN).read().to_pylist():
        sol, _, ans = r["answer"].rpartition("####")
        out.append({"problem": r["question"],
                    "generated_solution": CALC.sub("", sol).strip(),
                    "expected_answer": ans.strip(),
                    "problem_source": "gsm8k_gold",
                    "reasoning": CALC.sub("", sol).strip()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--max-total", type=int, default=0, help="0 = no cap")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="train_1M")
    ap.add_argument("--model", default="/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/"
                                       "snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)

    gold = load_gsm8k_train()
    omi = load_omi2(args.splits.split(","))
    print(f"gsm8k gold train: {len(gold)}  omi2 gsm8k-derived: {len(omi)}", flush=True)

    # few-shot pool: gold train examples formatted exactly as sample_to_fewshot does
    shot_pool = [(g["problem"], g["reasoning"], g["expected_answer"]) for g in gold]

    per_problem: dict[str, int] = {}
    seen_pair: set[tuple[str, str]] = set()
    kept = []
    dropped = {"answer": 0, "dup": 0, "cap": 0, "len": 0, "short": 0}

    pool = gold + omi
    rng.shuffle(pool)
    for r in pool:
        ans = norm_answer(r["expected_answer"])
        if ans is None:
            dropped["answer"] += 1
            continue
        sol = unbox(r["generated_solution"]).strip()
        sol = CALC.sub("", sol).strip()
        if len(sol) < 30:
            dropped["short"] += 1
            continue
        q = r["problem"].strip()
        key = (q, sol)
        if key in seen_pair:
            dropped["dup"] += 1
            continue
        cap = 3 if r["problem_source"] == "gsm8k_gold" else args.max_per_problem
        if per_problem.get(q, 0) >= cap:
            dropped["cap"] += 1
            continue
        seen_pair.add(key)
        per_problem[q] = per_problem.get(q, 0) + 1
        kept.append({"question": q, "solution": sol, "answer": ans,
                     "src": r["problem_source"]})

    print("kept", len(kept), "dropped", dropped, flush=True)
    rng.shuffle(kept)
    if args.max_total:
        kept = kept[: args.max_total]

    n_long = 0
    with open(args.out, "w") as f:
        for i, r in enumerate(kept):
            system = None
            if rng.random() < args.fewshot_frac:
                k = rng.randint(1, 8)
                system = render.fewshot_system_message(rng.sample(shot_pool, k))
            prompt = render.render_prompt(tok, r["question"], system)
            completion = render.render_completion(r["solution"], r["answer"])
            n = len(tok(prompt + completion, add_special_tokens=False)["input_ids"])
            if n > args.max_tokens:
                n_long += 1
                continue
            f.write(json.dumps({"prompt": prompt, "completion": completion,
                                "question": r["question"], "answer": r["answer"],
                                "src": r["src"], "n_tokens": n}) + "\n")
    print(f"wrote {args.out}; dropped {n_long} rows over {args.max_tokens} tokens", flush=True)


if __name__ == "__main__":
    main()
