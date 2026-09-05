#!/usr/bin/env python3
"""Build the SFT dataset for GSM8K-style math word problems.

Sources (all GSM8K *train*-derived or independent; never the test split):
  - openai/gsm8k train split (original human solutions)
  - nvidia/OpenMathInstruct-2, problem_source in {gsm8k, augmented_gsm8k}
    (both are derived from the GSM8K *train* split only)

Output: data/sft.jsonl  with fields {"question", "solution", "answer"}
"""
import argparse, json, os, random, re, sys
from collections import defaultdict

import pyarrow.parquet as pq
from datasets import load_dataset

random.seed(1234)

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
CALC = re.compile(r"<<[^>]*>>")


def norm_q(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def fmt_answer(a: str) -> str | None:
    """Normalise an expected answer to the string we want the model to emit."""
    a = a.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    if a.startswith("\\") or not a:
        return None
    try:
        f = float(a)
    except ValueError:
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    if abs(f - round(f)) < 1e-9:
        n = int(round(f))
        if abs(n) >= 10**12:
            return None
        return f"{n:,}"
    if abs(f) >= 10**12:
        return None
    s = ("%.6f" % f).rstrip("0").rstrip(".")
    return s


def clean_solution(sol: str, ans_disp: str) -> str | None:
    s = sol.strip()
    # unwrap \boxed{...}
    s = BOXED.sub(r"\1", s)
    if "\\boxed" in s:
        return None
    s = CALC.sub("", s)
    s = s.replace("\\[", "").replace("\\]", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    if not s:
        return None
    return s + "\n\nANSWER: " + ans_disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--n-omi-gsm8k", type=int, default=3)   # solutions per orig train problem
    ap.add_argument("--n-omi-aug", type=int, default=2)     # solutions per augmented problem
    ap.add_argument("--max-aug-problems", type=int, default=22000)
    args = ap.parse_args()

    # ---- test-set questions, used only to *exclude* overlapping training items ----
    test = json.load(open("/home/ben/test_data.json"))
    test_norm = {norm_q(x["question"]) for x in test}
    print("test questions:", len(test_norm))

    gs = load_dataset("openai/gsm8k", "main")
    train_norm = {norm_q(q) for q in gs["train"]["question"]}
    print("train questions:", len(train_norm), "overlap w/ test:", len(train_norm & test_norm))

    records = []
    dropped_contam = 0

    # ---------- 1. original GSM8K train solutions ----------
    for q, a in zip(gs["train"]["question"], gs["train"]["answer"]):
        if norm_q(q) in test_norm:
            dropped_contam += 1
            continue
        reasoning, _, final = a.partition("####")
        disp = fmt_answer(final)
        if disp is None:
            continue
        sol = clean_solution(reasoning, disp)
        if sol is None:
            continue
        records.append({"question": q.strip(), "solution": sol, "answer": disp,
                        "src": "gsm8k_orig"})
    print("after gsm8k_orig:", len(records))

    # ---------- 2. OpenMathInstruct-2 ----------
    def add_omi(path, per_problem, max_problems, tag):
        t = pq.read_table(path).to_pylist()
        random.shuffle(t)
        by_prob = defaultdict(list)
        for r in t:
            by_prob[r["problem"].strip()].append(r)
        keys = list(by_prob.keys())
        random.shuffle(keys)
        if max_problems:
            keys = keys[:max_problems]
        nonlocal dropped_contam
        n0 = len(records)
        for k in keys:
            if norm_q(k) in test_norm:
                dropped_contam += 1
                continue
            rows = by_prob[k]
            # prefer medium-length solutions (short ones skip steps, long ones ramble)
            rows.sort(key=lambda r: abs(len(r["generated_solution"]) - 550))
            kept = 0
            seen_sol = set()
            for r in rows:
                if kept >= per_problem:
                    break
                disp = fmt_answer(r["expected_answer"])
                if disp is None:
                    continue
                sol = r["generated_solution"]
                if len(sol) < 60 or len(sol) > 2200:
                    continue
                cs = clean_solution(sol, disp)
                if cs is None or cs in seen_sol:
                    continue
                seen_sol.add(cs)
                records.append({"question": k, "solution": cs, "answer": disp, "src": tag})
                kept += 1
        print(f"after {tag}: {len(records)} (+{len(records)-n0})")

    add_omi("data/omi2_gsm8k.parquet", args.n_omi_gsm8k, None, "omi_gsm8k")
    add_omi("data/omi2_augmented_gsm8k.parquet", args.n_omi_aug,
            args.max_aug_problems, "omi_aug")

    print("dropped for test overlap:", dropped_contam)
    random.shuffle(records)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(records), "->", args.out)
    from collections import Counter
    print(Counter(r["src"] for r in records))


if __name__ == "__main__":
    main()
