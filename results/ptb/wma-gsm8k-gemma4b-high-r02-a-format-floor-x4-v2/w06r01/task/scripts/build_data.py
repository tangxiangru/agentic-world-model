#!/usr/bin/env python3
"""Build the SFT pool for GSM8K from OpenMathInstruct-2 (gsm8k-derived rows only)
plus the original GSM8K train split.

Target format is exactly what the grader expects:
  user      : inspect_evals gsm8k MATH_PROMPT_TEMPLATE with the question
  assistant : step-by-step reasoning, then a final line "ANSWER: <number>"

The grader is inspect_ai match(numeric=True, location="end"): it takes the LAST
numeric token of the completion. So the ANSWER line must be last and must hold
nothing but the number.

Nothing here reads the GSM8K test split.
"""
import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq

TASK = "/home/ben/task"
OMI2_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"
NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")
BOXED = re.compile(r"\\boxed\{")


def strip_boxed(sol: str) -> str | None:
    """Replace every \\boxed{X} with X (brace-balanced). None if unbalanced."""
    out = []
    i = 0
    while True:
        m = BOXED.search(sol, i)
        if not m:
            out.append(sol[i:])
            break
        out.append(sol[i:m.start()])
        j = m.end()
        depth = 1
        while j < len(sol) and depth:
            if sol[j] == "{":
                depth += 1
            elif sol[j] == "}":
                depth -= 1
            j += 1
        if depth:
            return None
        out.append(sol[m.end():j - 1])
        i = j
    return "".join(out)


def clean_solution(sol: str, ans: str) -> str | None:
    s = strip_boxed(sol)
    if s is None:
        return None
    s = s.replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")
    s = re.sub(r"[ \t]+\n", "\n", s).strip()
    if not s:
        return None
    # drop a dangling "The answer is." style tail with nothing after it
    return f"{s}\n\nANSWER: {ans}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{TASK}/data/pool.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prompt_tpl = json.load(open(f"{TASK}/data/eval_prompt.json"))["prompt_template"]
    rng = random.Random(args.seed)

    # ---- 1. OpenMathInstruct-2, gsm8k + augmented_gsm8k only -----------------
    per_problem: dict[str, list[dict]] = defaultdict(list)
    files = sorted(glob.glob(OMI2_GLOB))
    assert files, OMI2_GLOB
    for f in files:
        t = pq.read_table(f, columns=["problem", "generated_solution",
                                      "expected_answer", "problem_source"])
        for r in t.to_pylist():
            if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                continue
            ans = (r["expected_answer"] or "").strip()
            if not NUMERIC.match(ans):
                continue
            body = clean_solution(r["generated_solution"], ans)
            if body is None or "ANSWER:" in r["generated_solution"]:
                continue
            per_problem[r["problem"].strip()].append(
                {"question": r["problem"].strip(), "target": body, "answer": ans,
                 "src": f"omi2:{r['problem_source']}"}
            )
        print(f"  read {os.path.basename(f)}: {len(per_problem)} unique problems", flush=True)

    rows = []
    for q, cands in per_problem.items():
        rng.shuffle(cands)
        # prefer the shortest solutions - they are the least likely to ramble
        cands.sort(key=lambda c: len(c["target"]))
        rows.extend(cands[: args.max_per_problem])

    # ---- 2. original GSM8K train split -------------------------------------
    import datasets
    gsm = datasets.load_dataset("openai/gsm8k", "main", split="train")
    for r in gsm:
        reasoning, _, ans = r["answer"].rpartition("####")
        ans = ans.strip()
        if not NUMERIC.match(ans):
            continue
        body = reasoning.strip()
        rows.append({"question": r["question"].strip(),
                     "target": f"{body}\n\nANSWER: {ans}",
                     "answer": ans, "src": "gsm8k:train"})

    rng.shuffle(rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_bad = 0
    with open(args.out, "w") as fh:
        for r in rows:
            # invariant: exactly one ANSWER marker, and the answer is the last number
            if r["target"].count("ANSWER:") != 1:
                n_bad += 1
                continue
            tail = r["target"].rsplit("ANSWER:", 1)[1].strip()
            if tail != r["answer"]:
                n_bad += 1
                continue
            fh.write(json.dumps({
                "question": r["question"],
                "prompt": prompt_tpl.replace("{prompt}", r["question"]),
                "target": r["target"],
                "answer": r["answer"],
                "src": r["src"],
            }) + "\n")
    print(f"wrote {len(rows) - n_bad} rows to {args.out} (dropped {n_bad})")


if __name__ == "__main__":
    main()
