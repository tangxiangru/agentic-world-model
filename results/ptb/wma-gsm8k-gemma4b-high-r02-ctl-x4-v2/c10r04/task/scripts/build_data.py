"""Build the SFT corpus for GSM8K.

Sources (both derived from GSM8K's *training* split only):
  1. openai/gsm8k main/train, minus a 300-item holdout (data/holdout_idx.json)
     -- the human-written reference solutions.
  2. nvidia/OpenMathInstruct-2 (train_1M), rows whose problem_source is
     gsm8k / augmented_gsm8k -- solutions written by Llama-3.1-405B-Instruct
     for GSM8K train problems and augmentations of them.

Every row is rendered with the grader's own prompt template (scripts/fmt.py) so
training and grading see byte-identical strings.  A fraction of rows carry the
grader's exact 10-shot system message so the model is also trained in the
context it is graded in.

Output jsonl fields:
  question    - raw problem text            (read by ../contamination_check.py)
  answer      - raw solution text           (read by ../contamination_check.py)
  prompt      - rendered prompt, ends with '<start_of_turn>model\\n'
  completion  - rendered target, ends with '<end_of_turn>'
  fewshot     - bool
  src         - 'gsm8k_gold' | 'omi2'
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

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
CALC_RE = re.compile(r"<<[^>]*>>")
INT_RE = re.compile(r"-?\d{1,12}")


def is_int_answer(s: str) -> bool:
    return bool(re.fullmatch(r"-?\d{1,9}", s.strip()))


def delatex(text: str) -> str | None:
    """Turn an OpenMathInstruct-2 body into plain prose+arithmetic, or None."""
    t = text.replace("\\$", "\x00")
    t = t.replace("$", "")
    t = t.replace("\x00", "$")
    t = t.replace("\\%", "%").replace("\\,", " ")
    if "\\" in t or "{" in t or "}" in t:
        return None
    return t


def clean_omi(sol: str, expected: str) -> str | None:
    sol = sol.strip()
    if sol.count("\\boxed{") != 1:
        return None
    body = BOXED_RE.sub(r"\1", sol)
    body = delatex(body)
    if body is None:
        return None
    body = body.strip()
    if not body or "ANSWER:" in body:
        return None
    return f"{body}\n\nANSWER: {expected}"


def clean_gold(answer: str) -> tuple[str, str] | None:
    if "####" not in answer:
        return None
    reasoning, final = answer.rsplit("####", 1)
    final = final.strip().replace(",", "")
    if not is_int_answer(final):
        return None
    body = CALC_RE.sub("", reasoning).strip()
    if not body or "ANSWER:" in body:
        return None
    return f"{body}\n\nANSWER: {final}", final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omi-dir", default="/home/ben/task/data/omi2_1M")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-omi", type=int, default=33000)
    ap.add_argument("--use-gold", type=int, default=1)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.2)
    ap.add_argument("--max-sol-chars", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []

    # ---- 1. gsm8k train gold, minus holdout ---------------------------------
    if args.use_gold:
        from datasets import load_dataset

        g = load_dataset("openai/gsm8k", "main")["train"]
        hold = set(json.load(open("/home/ben/task/data/holdout_idx.json")))
        n_gold = 0
        for i in range(len(g)):
            if i in hold:
                continue
            out = clean_gold(g[i]["answer"])
            if out is None:
                continue
            rows.append({"question": g[i]["question"].strip(), "answer": out[0], "src": "gsm8k_gold"})
            n_gold += 1
        print(f"gsm8k gold kept {n_gold} (holdout {len(hold)} excluded)")

    # ---- 2. OpenMathInstruct-2 ---------------------------------------------
    from datasets import load_from_disk

    ds = load_from_disk(args.omi_dir)
    keep_sources = {"gsm8k", "augmented_gsm8k"}
    order = list(range(len(ds)))
    rng.shuffle(order)
    per_problem: dict[str, int] = {}
    n_src = n_rej = n_omi = 0
    for i in order:
        if n_omi >= args.n_omi:
            break
        r = ds[i]
        if r["problem_source"] not in keep_sources:
            continue
        n_src += 1
        exp = str(r["expected_answer"]).strip().replace(",", "")
        if not is_int_answer(exp):
            continue
        prob = r["problem"].strip()
        if per_problem.get(prob, 0) >= args.max_per_problem:
            continue
        sol = clean_omi(r["generated_solution"], exp)
        if sol is None:
            n_rej += 1
            continue
        if len(sol) > args.max_sol_chars or len(prob) > 1200:
            continue
        # the grader reads the LAST number in the response; make sure it is ours
        if INT_RE.findall(sol)[-1].lstrip("-") != exp.lstrip("-"):
            continue
        per_problem[prob] = per_problem.get(prob, 0) + 1
        rows.append({"question": prob, "answer": sol, "src": "omi2"})
        n_omi += 1
    print(f"omi2: scanned {n_src} gsm8k-sourced rows, rejected-format {n_rej}, kept {n_omi}")

    rng.shuffle(rows)
    n_few = int(round(args.fewshot_frac * len(rows)))
    with open(args.out, "w") as f:
        for k, row in enumerate(rows):
            few = k < n_few
            rec = {
                "question": row["question"],
                "answer": row["answer"],
                "prompt": fmt.render_prompt(row["question"], fewshot=few),
                "completion": fmt.render_target(row["answer"]),
                "fewshot": few,
                "src": row["src"],
            }
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} ({n_few} with the 10-shot system message)")


if __name__ == "__main__":
    main()
