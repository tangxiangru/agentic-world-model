#!/usr/bin/env python3
"""Build the SFT file for GSM8K.

Sources (both derived only from *training* splits, never from the benchmark
test split):
  * nvidia/OpenMathInstruct-2, rows with problem_source in {gsm8k,
    augmented_gsm8k} -- solutions written by Llama-3.1-405B-Instruct for GSM8K
    *train* problems and for augmentations of them.
  * openai/gsm8k main/train original human solutions, minus the 300-item local
    holdout in data/dev_train_holdout.jsonl.

Every target is shaped for the grader: chain of thought, then a final line
"ANSWER: <number>", nothing after it. The grader is
inspect_ai match(numeric=True, location="end"), which reads the last numeric
whitespace token of the completion.
"""
import argparse
import json
import os
import random
import re

import pandas as pd

OMI_DIR = ("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
           "snapshots/469216e3f46f4dacf476b382e192485ea51a143e/data")

BOXED_RE = re.compile(r"\\boxed\s*\{")
FINAL_SENT_RE = re.compile(
    r"(?:\n+|\s)(?:so\s+)?the\s+(?:final\s+)?answer\s+is[^\n]*$", re.IGNORECASE)
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")

# the terminator templates/gemma3.jinja emits; every target must end with it
END_OF_TURN = "<end_of_turn>"


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
    while True:
        m = BOXED_RE.search(text)
        if not m:
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
        text = text[: m.start()] + text[start: i - 1] + text[i:]


def normalise_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def clean_answer(a: str) -> str | None:
    a = str(a).strip().replace(",", "").replace("$", "").replace("%", "")
    a = a.rstrip(".")
    if not NUM_RE.match(a):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    return a


def make_target(solution: str, answer: str) -> str | None:
    s = strip_boxed(solution).strip()
    s = FINAL_SENT_RE.sub("", s).strip()
    s = s.rstrip("$ \n\t.")
    if not s or "ANSWER:" in s.upper():
        return None
    if len(s) < 20:
        return None
    return f"{s}\n\nANSWER: {answer}{END_OF_TURN}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--per-problem", type=int, default=2)
    ap.add_argument("--max-omi", type=int, default=60000)
    ap.add_argument("--gsm8k-repeat", type=int, default=1)
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    holdout = {normalise_q(json.loads(l)["question"])
               for l in open("data/dev_train_holdout.jsonl")}
    print(f"holdout questions: {len(holdout)}")

    rows = []

    # ---- OpenMathInstruct-2, GSM8K-derived rows -----------------------------
    seen_per_problem: dict[str, int] = {}
    kept = dropped_ans = dropped_sol = dropped_holdout = 0
    for i in range(args.shards):
        path = os.path.join(OMI_DIR, f"train-{i:05d}-of-00032.parquet")
        df = pd.read_parquet(
            path, columns=["problem", "generated_solution", "expected_answer",
                           "problem_source"])
        df = df[df.problem_source.isin(["gsm8k", "augmented_gsm8k"])]
        for prob, sol, ans in zip(df.problem, df.generated_solution,
                                  df.expected_answer):
            key = normalise_q(prob)
            if key in holdout:
                dropped_holdout += 1
                continue
            if seen_per_problem.get(key, 0) >= args.per_problem:
                continue
            a = clean_answer(ans)
            if a is None:
                dropped_ans += 1
                continue
            t = make_target(sol, a)
            if t is None:
                dropped_sol += 1
                continue
            seen_per_problem[key] = seen_per_problem.get(key, 0) + 1
            rows.append({"question": prob.strip(), "target": t,
                         "answer": a, "src": "omi2"})
            kept += 1
        print(f"shard {i}: running kept={kept}", flush=True)
        if kept >= args.max_omi:
            break
    rng.shuffle(rows)
    rows = rows[: args.max_omi]
    print(f"omi2 kept={len(rows)} dropped_ans={dropped_ans} "
          f"dropped_sol={dropped_sol} dropped_holdout={dropped_holdout} "
          f"distinct_problems={len(seen_per_problem)}")

    # ---- original GSM8K train solutions -------------------------------------
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")["train"]
    n_gsm = 0
    for r in ds:
        if normalise_q(r["question"]) in holdout:
            continue
        body, _, ans = r["answer"].rpartition("####")
        a = clean_answer(ans)
        if a is None:
            continue
        # drop the calculator annotations: the model cannot execute them
        body = re.sub(r"<<[^>]*>>", "", body).strip()
        t = f"{body}\n\nANSWER: {a}{END_OF_TURN}"
        for _ in range(args.gsm8k_repeat):
            rows.append({"question": r["question"].strip(), "target": t,
                         "answer": a, "src": "gsm8k_train"})
        n_gsm += 1
    print(f"gsm8k_train kept={n_gsm} (x{args.gsm8k_repeat})")

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
