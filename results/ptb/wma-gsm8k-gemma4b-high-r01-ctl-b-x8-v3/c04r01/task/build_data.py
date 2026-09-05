#!/usr/bin/env python3
"""Build the SFT corpus for GSM8K post-training of gemma-3-4b-pt.

Every row is written as {"prompt": <rendered user turn>, "completion": <target>}
where the strings are already in the grader's chat format:

    prompt     = "<bos><start_of_turn>user\n{...}<end_of_turn>\n<start_of_turn>model\n"
    completion = "{solution}\nANSWER: {n}<end_of_turn>"

The prompt body reproduces inspect_evals.gsm8k.MATH_PROMPT_TEMPLATE byte for byte
and the wrapper reproduces templates/gemma3.jinja, so training and grading render
the same string for the same question.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re

import pandas as pd

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

HERE = os.path.dirname(os.path.abspath(__file__))

# --- byte-for-byte copy of inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE ---
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

GSM8K_TRAIN = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet"
OMI2_DIR = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/469216e3f46f4dacf476b382e192485ea51a143e/data"

CALC = re.compile(r"<<[^>]*>>")
BOXED = re.compile(r"\\boxed\{")
NUMERIC = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{X} with X (brace-balanced)."""
    out = []
    i = 0
    while True:
        m = BOXED.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i : m.start()])
        j = m.end()
        depth = 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append(text[m.end() : j])
        i = j + 1
    return "".join(out)


def norm_num(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUMERIC.match(s):
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


def render_prompt(question: str, sys_prefix: str = "") -> str:
    body = MATH_PROMPT_TEMPLATE.format(prompt=question).strip()
    return (
        "<bos><start_of_turn>user\n"
        + sys_prefix
        + body
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def fewshot_block(q: str, reasoning: str, ans: str) -> str:
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {ans}"


def load_gsm8k_train() -> list[dict]:
    df = pd.read_parquet(GSM8K_TRAIN)
    rows = []
    for q, a in zip(df["question"], df["answer"]):
        head, _, tail = a.rpartition("####")
        ans = norm_num(tail)
        if ans is None:
            continue
        body = CALC.sub("", head).strip()
        rows.append({"question": q.strip(), "solution": body, "answer": ans,
                     "src": "gsm8k_train"})
    return rows


def load_omi2(shards: list[int], sources: set[str]) -> list[dict]:
    rows = []
    for i in shards:
        p = os.path.join(OMI2_DIR, f"train-{i:05d}-of-00032.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p, columns=["problem", "generated_solution",
                                         "expected_answer", "problem_source"])
        df = df[df["problem_source"].isin(sources)]
        for q, s, a, src in zip(df["problem"], df["generated_solution"],
                                df["expected_answer"], df["problem_source"]):
            ans = norm_num(str(a))
            if ans is None:
                continue
            sol = strip_boxed(s).strip()
            rows.append({"question": q.strip(), "solution": sol, "answer": ans,
                         "src": src})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data/sft_v1.jsonl"))
    ap.add_argument("--shards", default="0,1,2,3,4,5")
    ap.add_argument("--max-per-problem", type=int, default=1)
    ap.add_argument("--sol-start", type=int, default=0)
    ap.add_argument("--exclude", default="", help="comma-separated jsonl files whose (question, completion) pairs must not reappear")
    ap.add_argument("--target-rows", type=int, default=140000)
    ap.add_argument("--fewshot-frac", type=float, default=0.05)
    ap.add_argument("--fewshot-min", type=int, default=2)
    ap.add_argument("--fewshot-max", type=int, default=5)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--holdout", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    shards = [int(x) for x in args.shards.split(",") if x != ""]

    # rows already used by an earlier card: the per-problem solution index is
    # only disjoint across builds when the shuffle seed matches, so exclude by
    # content instead of trusting the index
    used: set = set()
    for f in [x for x in args.exclude.split(",") if x]:
        for line in open(f):
            d = json.loads(line)
            used.add((d["question"], d["completion"].split("\n\nANSWER:")[0]))
    if used:
        print(f"excluding {len(used)} (question, solution) pairs already used")

    ref = load_gsm8k_train()
    print(f"gsm8k train reference rows: {len(ref)}")

    # held-out probe set: gsm8k TRAIN items never shown to the trainer.
    # (the benchmark's own test split is only ever input to the contamination
    # checker, so diagnostics need their own dev set)
    holdout = ref[-args.holdout:] if args.holdout else []
    ref = ref[: len(ref) - len(holdout)]
    hold_qs = {h["question"] for h in holdout}
    if holdout:
        dev_path = os.path.join(HERE, "data/dev_train250.jsonl")
        with open(dev_path, "w") as f:
            for i, h in enumerate(holdout):
                f.write(json.dumps({"id": f"devtrain-{i:03d}",
                                    "question": h["question"],
                                    "gold": h["answer"]}) + "\n")
        print(f"held out {len(holdout)} gsm8k-train items -> {dev_path}")
    omi = load_omi2(shards, {"gsm8k", "augmented_gsm8k"})
    print(f"omi2 gsm8k-flavoured rows: {len(omi)}")

    # ---- dedup / cap solutions per problem -----------------------------------
    seen: dict[str, int] = {}
    kept = []
    rng.shuffle(omi)
    for r in omi:
        if r["question"] in hold_qs:
            continue
        if used and (r["question"], r["solution"].strip()) in used:
            continue
        k = hashlib.md5(r["question"].encode()).hexdigest()
        c = seen.get(k, 0)
        seen[k] = c + 1
        if c < args.sol_start or c >= args.sol_start + args.max_per_problem:
            continue
        kept.append(r)
    print(f"after dedup (<= {args.max_per_problem}/problem): {len(kept)}")

    rng.shuffle(kept)
    kept = kept[: max(0, args.target_rows - len(ref) * args.gsm8k_repeat)]
    rows = kept + ref * args.gsm8k_repeat
    rng.shuffle(rows)
    print(f"total rows: {len(rows)}")

    # ---- few-shot prefix pool (from gsm8k TRAIN only) -------------------------
    pool = [fewshot_block(r["question"], r["solution"], r["answer"])
            for r in ref[:2000]]

    n_written = 0
    n_skipped = 0
    with open(args.out, "w") as f:
        for r in rows:
            sol = r["solution"].strip()
            if not sol or len(sol) > 4000:
                n_skipped += 1
                continue
            # one answer marker only: drop any trailing "The answer is ..." line
            completion = f"{sol}\n\nANSWER: {r['answer']}<end_of_turn>"
            if completion.count("ANSWER:") != 1:
                n_skipped += 1
                continue
            sys_prefix = ""
            if rng.random() < args.fewshot_frac:
                k = rng.randint(args.fewshot_min, args.fewshot_max)
                sys_prefix = "\n\n".join(rng.sample(pool, k)) + "\n\n"
            f.write(json.dumps({
                "prompt": render_prompt(r["question"], sys_prefix),
                "completion": completion,
                "question": r["question"],
                "answer": r["answer"],
                "src": r["src"],
            }) + "\n")
            n_written += 1
    print(f"wrote {n_written} rows to {args.out} (skipped {n_skipped})")


if __name__ == "__main__":
    main()
