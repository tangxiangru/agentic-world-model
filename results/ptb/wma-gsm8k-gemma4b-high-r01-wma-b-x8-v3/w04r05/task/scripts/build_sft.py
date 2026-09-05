#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt -> GSM8K.

Sources (all GSM8K *train* / MATH train derived -- the benchmark test split is
never read here):
  * nvidia/OpenMathInstruct-2 train_1M, rows with problem_source in
    {gsm8k, augmented_gsm8k}   -> the bulk
  * the same file's {math, augmented_math} rows whose expected_answer is a plain
    number                     -> harder arithmetic, keeps the answer format uniform
  * openai/gsm8k train split gold solutions -> style anchor: these are written in
    exactly the register of the 10-shot exemplars the grader puts in the system
    message (short lines, <<a*b=c>> annotations).

Output rows: {"prompt": <rendered up to '<start_of_turn>model\\n'>,
              "completion": <solution + '\\n\\nANSWER: N' + '<end_of_turn>\\n'>,
              "answer": N, "src": ...}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    assert_template_matches_grader,
    is_plain_int,
    render_completion,
    render_prompt,
    sample_to_fewshot,
)


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{X} / \\boxed X with X (brace-matched)."""
    out = []
    i = 0
    while True:
        j = text.find("\\boxed", i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        k = j + len("\\boxed")
        while k < len(text) and text[k] == " ":
            k += 1
        if k < len(text) and text[k] == "{":
            depth = 0
            m = k
            while m < len(text):
                if text[m] == "{":
                    depth += 1
                elif text[m] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                m += 1
            out.append(text[k + 1 : m])
            i = m + 1
        else:
            i = k


_MATHY = re.compile(r"\\(frac|sqrt|begin|pmatrix|int|sum|lim|cdot|times|div|le|ge|neq|approx|pi|theta)")


def clean_solution(sol: str) -> str:
    s = strip_boxed(sol)
    # drop stray latex display delimiters that now wrap a bare number
    s = re.sub(r"\\\[\s*", "", s)
    s = re.sub(r"\s*\\\]", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm8k-aug", type=int, default=100000)
    ap.add_argument("--n-math", type=int, default=25000)
    ap.add_argument("--gold-repeats", type=int, default=3)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=500, help="gsm8k train problems reserved as a dev set")
    ap.add_argument("--holdout-out", default="data/dev_train500.jsonl")
    args = ap.parse_args()

    assert_template_matches_grader()
    rng = random.Random(args.seed)

    # ---- gsm8k train gold -------------------------------------------------
    from datasets import load_dataset

    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    gold = []
    for r in gsm_train:
        q = r["question"].strip()
        reasoning, ans = r["answer"].split("####")
        gold.append({"question": q, "reasoning": reasoning.strip(), "answer": ans.strip()})
    rng.shuffle(gold)

    dev = gold[: args.holdout]
    gold = gold[args.holdout :]
    with open(args.holdout_out, "w") as f:
        for i, d in enumerate(dev):
            f.write(json.dumps({"id": f"devtrain-{i:04d}", "question": d["question"], "gold": d["answer"]}) + "\n")
    held_q = {d["question"] for d in dev}
    print(f"held out {len(dev)} gsm8k-train problems -> {args.holdout_out}", flush=True)

    # exemplar pool for the few-shot-prefix augmentation (gold, train split only)
    fewshot_pool = gold[:2000]

    rows = []

    def add(prompt_q, solution, answer, src):
        sol = clean_solution(solution)
        if not sol:
            return
        body = f"{sol}\n\nANSWER: {answer}"
        system = None
        if rng.random() < args.fewshot_frac:
            k = rng.randint(2, 8)
            shots = rng.sample(fewshot_pool, k)
            system = "\n\n".join(
                sample_to_fewshot(s["question"], s["reasoning"], s["answer"]) for s in shots
            )
        rows.append(
            {
                "prompt": render_prompt(prompt_q, system),
                "completion": render_completion(body),
                "answer": answer,
                "src": src,
            }
        )

    # ---- OMI-2 gsm8k-derived ---------------------------------------------
    by_problem = defaultdict(list)
    with open("data/omi2_raw_gsm8k.jsonl") as f:
        for line in f:
            r = json.loads(line)
            a = (r["expected_answer"] or "").strip()
            if not is_plain_int(a):
                continue
            if r["problem"].strip() in held_q:
                continue  # never train on a problem we score ourselves on
            by_problem[r["problem"].strip()].append(r)
    probs = sorted(by_problem)
    rng.shuffle(probs)
    n = 0
    for p in probs:
        cands = by_problem[p]
        rng.shuffle(cands)
        for r in cands[: args.max_per_problem]:
            if n >= args.n_gsm8k_aug:
                break
            add(p, r["generated_solution"], r["expected_answer"].strip(), r["problem_source"])
            n += 1
        if n >= args.n_gsm8k_aug:
            break
    print(f"omi2 gsm8k rows: {n} (from {len(probs)} distinct problems)", flush=True)

    # ---- OMI-2 math-derived, numeric answers only -------------------------
    by_problem = defaultdict(list)
    with open("data/omi2_raw_math.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if is_plain_int((r["expected_answer"] or "").strip()):
                by_problem[r["problem"].strip()].append(r)
    probs = sorted(by_problem)
    rng.shuffle(probs)
    m = 0
    for p in probs:
        r = by_problem[p][0]
        if m >= args.n_math:
            break
        if _MATHY.search(r["problem"]):
            continue  # keep the corpus in word-problem register
        add(p, r["generated_solution"], r["expected_answer"].strip(), "math")
        m += 1
    print(f"omi2 math rows: {m}", flush=True)

    # ---- gsm8k train gold, repeated ---------------------------------------
    g = 0
    for _ in range(args.gold_repeats):
        for d in gold:
            add(d["question"], d["reasoning"], d["answer"], "gsm8k_gold")
            g += 1
    print(f"gsm8k gold rows: {g}", flush=True)

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
