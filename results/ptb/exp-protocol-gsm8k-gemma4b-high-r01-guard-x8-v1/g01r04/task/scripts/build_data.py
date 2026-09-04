#!/usr/bin/env python3
"""Build SFT data for GSM8K in the grader's exact answer format.

Sources (both derived from the GSM8K *train* split or from augmentations of it;
the test split is never read here):
  * openai/gsm8k, split=train  -- original human solutions
  * nvidia/OpenMathInstruct-2, split=train_1M, problem_source in
    {gsm8k, augmented_gsm8k} -- Llama-3.1-405B solutions to GSM8K train
    problems and to LLM-augmented variants of them

Output rows: {"question", "completion", "system_mode"} where `completion` is the
literal model turn including the terminator the grading template stops on.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

CALC = re.compile(r"<<[^>]*>>")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
INT_RE = re.compile(r"^-?\d+$")


def clean_int(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    if INT_RE.match(s):
        return str(int(s))
    return None


def finalize(body: str, answer: str) -> str | None:
    """Body + the single answer line, terminated by the stop token."""
    body = body.strip()
    if not body:
        return None
    if "ANSWER:" in body:
        return None
    if fmt.STOP_TOKEN in body or "<start_of_turn>" in body:
        return None
    return f"{body}\n\n{fmt.ANSWER_MARKER}{answer}{fmt.STOP_TOKEN}"


def gsm8k_rows(holdout_n: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    hold = set(idx[:holdout_n])
    train, dev = [], []
    for i, r in enumerate(ds):
        q = r["question"].strip()
        body, _, ans = r["answer"].rpartition("####")
        ans = clean_int(ans)
        if ans is None:
            continue
        body = CALC.sub("", body).strip()
        comp = finalize(body, ans)
        if comp is None:
            continue
        if i in hold:
            dev.append({"question": q, "answer": ans})
        else:
            train.append({"question": q, "completion": comp, "src": "gsm8k_train"})
    return train, dev


def omi_rows(dev_questions: set[str], max_per_problem: int):
    from datasets import load_dataset

    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    ds = ds.filter(
        lambda x: x["problem_source"] in ("gsm8k", "augmented_gsm8k"), num_proc=8
    )
    seen: dict[str, int] = {}
    out = []
    for r in ds:
        ans = clean_int(r["expected_answer"])
        if ans is None:
            continue
        q = r["problem"].strip()
        if q in dev_questions:
            continue
        n = seen.get(q, 0)
        if n >= max_per_problem:
            continue
        sol = r["generated_solution"]
        # keep the value, drop the \boxed wrapper so the prose still reads
        sol = BOXED.sub(lambda m: m.group(1), sol)
        if "\\boxed" in sol:
            continue
        comp = finalize(sol, ans)
        if comp is None:
            continue
        seen[q] = n + 1
        out.append({"question": q, "completion": comp, "src": r["problem_source"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--dev-out", default="data/dev_train500.jsonl")
    ap.add_argument("--holdout", type=int, default=500)
    ap.add_argument("--n-omi", type=int, default=48000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--n-fewshot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    g_train, dev = gsm8k_rows(args.holdout, args.seed)
    print(f"gsm8k train rows: {len(g_train)}  holdout dev: {len(dev)}", flush=True)

    dev_q = {d["question"] for d in dev}
    omi = omi_rows(dev_q, args.max_per_problem)
    print(f"omi gsm8k-family rows: {len(omi)}", flush=True)
    rng.shuffle(omi)
    omi = omi[: args.n_omi]

    rows = g_train + omi
    rng.shuffle(rows)
    # a slice gets the grader's exact 10-shot system prefix so the model has
    # seen that context shape terminate correctly
    for i, r in enumerate(rows):
        r["system_mode"] = "fewshot" if i < args.n_fewshot else "zeroshot"
    rng.shuffle(rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(args.dev_out, "w") as f:
        for i, d in enumerate(dev):
            d["id"] = f"trdev-{i:04d}"
            f.write(json.dumps(d) + "\n")
    print(f"wrote {len(rows)} -> {args.out}", flush=True)
    print(f"wrote {len(dev)} -> {args.dev_out}", flush=True)


if __name__ == "__main__":
    main()
