#!/usr/bin/env python3
"""Build the SFT corpus in exactly the shape the grader reads.

Every row is {question, completion, answer, source}:
  * `question`  the raw word problem (what goes into MATH_PROMPT_TEMPLATE)
  * `completion` reasoning + blank line + "ANSWER: <n>" + "<end_of_turn>"
  * `answer`    a copy of `completion` so ../contamination_check.py sees a target
  * `source`    provenance tag

Sources (all derived from GSM8K *train* or from synthetic problems; the GSM8K
test split is never read here):
  gsm8k_train        openai/gsm8k train split, reference solutions verbatim
  omi2_gsm8k         nvidia/OpenMathInstruct-2 train_1M, problem_source in
                     {gsm8k, augmented_gsm8k} (Llama-3.1-405B solutions to
                     GSM8K-train problems and to augmentations of them)
  metamath_gsm       meta-math/MetaMathQA GSM_* subsets (optional)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP_TOKEN = "<end_of_turn>"
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
CALC_RE = re.compile(r"<<[^>]*>>")


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    a = a.rstrip(".")
    if not NUM_RE.match(a):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    return a


def make_target(body: str, answer: str) -> str | None:
    body = body.strip()
    if not body:
        return None
    # one answer marker only (pitfall double_answer_format)
    body = re.sub(r"\n?####\s*[-\d.,$]+\s*", "\n", body)
    body = re.sub(r"\nThe answer is:.*$", "", body).strip()
    if "ANSWER:" in body or "####" in body:
        return None
    return f"{body}\n\nANSWER: {answer}{STOP_TOKEN}"


def load_gsm8k_train(exclude_questions: set[str]) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for rec in ds:
        q = rec["question"].strip()
        if q in exclude_questions:
            continue
        parts = rec["answer"].split("####")
        ans = norm_answer(parts[-1])
        body = "####".join(parts[:-1]).strip()
        if ans is None:
            continue
        tgt = make_target(body, ans)
        if tgt is None:
            continue
        out.append({"question": q, "completion": tgt, "source": "gsm8k_train", "final": ans})
    return out


def load_omi2(max_per_problem: int, limit: int | None, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    files = sorted(
        glob.glob(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
            "snapshots/*/data/train_1M-*.parquet"
        )
    )
    rows: list[dict] = []
    per_problem: dict[str, int] = defaultdict(int)
    for f in files:
        df = pq.read_table(f).to_pandas()
        df = df[df.problem_source.isin(["gsm8k", "augmented_gsm8k"])]
        for problem, sol, exp in zip(df.problem, df.generated_solution, df.expected_answer):
            ans = norm_answer(str(exp))
            if ans is None:
                continue
            if per_problem[problem] >= max_per_problem:
                continue
            body = BOXED_RE.sub(r"\1", str(sol))
            if "\\boxed" in body:
                continue
            tgt = make_target(body, ans)
            if tgt is None:
                continue
            per_problem[problem] += 1
            rows.append(
                {"question": str(problem).strip(), "completion": tgt,
                 "source": "omi2_gsm8k", "final": ans}
            )
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def load_metamath(limit: int | None, seed: int) -> list[dict]:
    path = glob.glob(
        "/home/ben/hf_cache/hub/datasets--meta-math--MetaMathQA/snapshots/*/MetaMathQA-395K.json"
    )
    if not path:
        return []
    data = json.load(open(path[0]))
    rows = []
    for rec in data:
        if not rec["type"].startswith("GSM"):
            continue
        resp = rec["response"]
        m = re.search(r"The answer is:\s*(.+)\s*$", resp)
        if not m:
            continue
        ans = norm_answer(m.group(1))
        if ans is None:
            continue
        body = resp[: m.start()]
        tgt = make_target(body, ans)
        if tgt is None:
            continue
        rows.append({"question": rec["query"].strip(), "completion": tgt,
                     "source": "metamath_gsm", "final": ans})
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--omi2", type=int, default=30000)
    ap.add_argument("--omi2-per-problem", type=int, default=1)
    ap.add_argument("--metamath", type=int, default=0)
    ap.add_argument("--gsm8k-repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fewshot_qs = set()
    p = os.path.join(TASK_DIR, "data", "eval_fewshot_questions.json")
    if os.path.exists(p):
        fewshot_qs = {q.strip() for q in json.load(open(p))}

    rows: list[dict] = []
    g = load_gsm8k_train(fewshot_qs)
    print(f"gsm8k_train: {len(g)}")
    rows += g * args.gsm8k_repeat
    if args.omi2:
        o = load_omi2(args.omi2_per_problem, args.omi2, args.seed)
        print(f"omi2_gsm8k: {len(o)}")
        rows += o
    if args.metamath:
        m = load_metamath(args.metamath, args.seed)
        print(f"metamath_gsm: {len(m)}")
        rows += m

    # exact-duplicate removal on (question, completion)
    seen = set()
    dedup = []
    for r in rows:
        k = (r["question"], r["completion"])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    random.Random(args.seed).shuffle(dedup)

    with open(args.out, "w") as f:
        for r in dedup:
            r = dict(r)
            r["answer"] = r["completion"]
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(dedup)} rows -> {args.out}")


if __name__ == "__main__":
    main()
