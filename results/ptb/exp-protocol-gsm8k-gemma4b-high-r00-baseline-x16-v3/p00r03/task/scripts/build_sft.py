"""Build the stage-1 SFT corpus.

Sources (all public HF datasets; the GSM8K *test* split is never read):
  * openai/gsm8k train split  -- human chain-of-thought, dev500 held out
  * nvidia/OpenMathInstruct-2 -- gsm8k / augmented_gsm8k subsets
  * meta-math/MetaMathQA      -- GSM-typed rows (optional)

Every target is normalised to: reasoning, blank line, then a final line
`ANSWER: <integer>` and nothing after it, because the grader
(inspect_ai match(numeric=True, location="end")) reads the LAST number in the
completion.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TASK_DIR, gsm8k_gold, strip_calc, user_prompt, write_jsonl  # noqa: E402

from datasets import load_dataset

INT_RE = re.compile(r"^-?\d+$")


def norm_answer(a):
    a = str(a).strip().replace(",", "").replace("$", "").rstrip(".")
    if a.endswith(".0"):
        a = a[:-2]
    return a


def is_int_answer(a):
    return bool(INT_RE.match(a))


def clean_solution_tail(sol: str) -> str:
    """Drop the source dataset's own answer announcement so ours is the only one."""
    sol = sol.strip()
    # OpenMathInstruct-2 / MetaMathQA announce the answer in prose
    cut = -1
    for marker in ["The final answer is", "The answer is:", "The answer is"]:
        i = sol.rfind(marker)
        if i != -1:
            cut = i if cut == -1 else min(cut, i)
    if cut != -1:
        sol = sol[:cut].strip()
    # otherwise the announcement is a \boxed{} display: drop that line onwards
    if "\\boxed" in sol:
        lines = sol.split("\n")
        for j in range(len(lines) - 1, -1, -1):
            if "\\boxed" in lines[j]:
                lines = lines[:j]
                break
        sol = "\n".join(lines).strip()
    # any leftover boxed inline
    sol = re.sub(r"\$?\\boxed\{([^}]*)\}\$?", r"\1", sol).strip()
    return sol


def make_row(question: str, solution: str, answer: str, src: str):
    solution = clean_solution_tail(solution)
    if not solution:
        return None
    target = f"{solution}\n\nANSWER: {answer}<end_of_turn>"   # the grader stops on this token
    return {
        "prompt": user_prompt(question),
        "completion": target,
        "question": question,
        "answer": answer,
        "source": src,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(TASK_DIR, "data", "sft_v1.jsonl"))
    ap.add_argument("--omi-max", type=int, default=110000)
    ap.add_argument("--omi-per-problem", type=int, default=2)
    ap.add_argument("--metamath-max", type=int, default=0)
    ap.add_argument("--gsm8k-repeat", type=int, default=1)
    ap.add_argument("--include-math", type=int, default=0, help="rows from OMI2 math subsets")
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(os.path.join(TASK_DIR, "data", "dev500_questions.json")) as f:
        dev_qs = set(json.load(f))
    dev_idx = set(json.load(open(os.path.join(TASK_DIR, "data", "dev500_train_idx.json"))))

    rows = []

    # ---- 1. GSM8K train, human solutions -----------------------------------
    g = load_dataset("openai/gsm8k", "main")["train"]
    n_gsm = 0
    for i in range(len(g)):
        if i in dev_idx:
            continue
        r = g[i]
        ans = norm_answer(gsm8k_gold(r["answer"]))
        if not is_int_answer(ans):
            continue
        body = strip_calc(r["answer"].split("####")[0]).strip()
        body = re.sub(r"[ \t]+\n", "\n", body)
        row = make_row(r["question"], body, ans, "gsm8k_train")
        if row:
            for _ in range(args.gsm8k_repeat):
                rows.append(dict(row))
            n_gsm += 1
    print(f"gsm8k_train: {n_gsm} problems -> {n_gsm * args.gsm8k_repeat} rows")

    # ---- 2. OpenMathInstruct-2 ---------------------------------------------
    if args.omi_max > 0 or args.include_math > 0:
        omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
        print("OMI2 columns:", omi.column_names)
        want_gsm = {"gsm8k", "augmented_gsm8k"}
        want_math = {"math", "augmented_math"}
        per_problem = {}
        gsm_rows, math_rows = [], []
        srcs = omi["problem_source"]
        for i, src in enumerate(srcs):
            if src in want_gsm and len(gsm_rows) < args.omi_max * 3:
                bucket = gsm_rows
            elif src in want_math and len(math_rows) < args.include_math * 3:
                bucket = math_rows
            else:
                continue
            bucket.append(i)
        rng.shuffle(gsm_rows)
        rng.shuffle(math_rows)

        def take(indices, cap, tag):
            out = []
            for i in indices:
                if len(out) >= cap:
                    break
                r = omi[i]
                q = r["problem"]
                if q in dev_qs:
                    continue
                if per_problem.get(q, 0) >= args.omi_per_problem:
                    continue
                ans = norm_answer(r["expected_answer"])
                if tag == "gsm" and not is_int_answer(ans):
                    continue
                sol = r["generated_solution"]
                if len(sol) > args.max_chars:
                    continue
                row = make_row(q, sol, ans, "omi2_" + r["problem_source"])
                if row is None:
                    continue
                per_problem[q] = per_problem.get(q, 0) + 1
                out.append(row)
            return out

        got = take(gsm_rows, args.omi_max, "gsm")
        rows += got
        print(f"omi2 gsm: {len(got)} rows")
        if args.include_math:
            gotm = take(math_rows, args.include_math, "math")
            rows += gotm
            print(f"omi2 math: {len(gotm)} rows")

    # ---- 3. MetaMathQA ------------------------------------------------------
    if args.metamath_max > 0:
        mm = load_dataset("meta-math/MetaMathQA", split="train")
        idxs = [i for i, t in enumerate(mm["type"]) if "GSM" in t]
        rng.shuffle(idxs)
        got = 0
        for i in idxs:
            if got >= args.metamath_max:
                break
            r = mm[i]
            q = r["query"]
            if q in dev_qs:
                continue
            resp = r["response"]
            m = re.search(r"The answer is:\s*([^\s]+)", resp)
            if not m:
                continue
            ans = norm_answer(m.group(1))
            if not is_int_answer(ans):
                continue
            row = make_row(q, resp, ans, "metamath_gsm")
            if row:
                rows.append(row)
                got += 1
        print(f"metamath: {got} rows")

    # ---- dedup + shuffle ----------------------------------------------------
    seen, dedup = set(), []
    for r in rows:
        k = (r["prompt"], r["completion"])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    rng.shuffle(dedup)
    write_jsonl(args.out, dedup)
    print(f"wrote {len(dedup)} rows -> {args.out}")
    from collections import Counter

    print(Counter(r["source"] for r in dedup))
    print("--- example ---")
    print(dedup[0]["prompt"][-300:])
    print("<<<COMPLETION>>>")
    print(dedup[0]["completion"])


if __name__ == "__main__":
    main()
