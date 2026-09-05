"""Build SFT jsonl in the exact format the grader expects.

Every row: {"question": ..., "completion": ..., "gold": ..., "src": ...}
`completion` is exactly what the trainer computes loss on: the chain of
thought, a final line "ANSWER: <number>", then the stop token <end_of_turn>
that the grading template stops on. Nothing is appended later.

Sources
  gsm8k   : openai/gsm8k main/train, indices 0..6972 (last 500 held out as probe)
  omi2    : nvidia/OpenMathInstruct-2 train_1M, problem_source in
            {gsm8k, augmented_gsm8k} (built from the GSM8K *train* split only)
  metamath: meta-math/MetaMathQA, type startswith GSM (built from GSM8K train)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

ANS_RE = re.compile(r"^-?\d+(\.\d+)?$")


def norm_num(s: str) -> str | None:
    s = str(s).strip().replace(",", "").replace("$", "").rstrip(".")
    if s.endswith("%"):
        s = s[:-1]
    if not ANS_RE.match(s):
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


def strip_boxed(text: str) -> str:
    """Remove \\boxed{...} wrappers, keeping their contents."""
    out = []
    i = 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k = j + len("\\boxed{")
        depth = 1
        while k < len(text) and depth:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        out.append(text[j + len("\\boxed{"):k])
        i = k + 1
    return "".join(out)


STOP = "<end_of_turn>"


def finalize(body: str, answer: str) -> str:
    body = body.strip()
    # drop any trailing "The answer is: X" style tail; we add our own marker
    body = re.sub(r"\n?The answer is:?\s*.*$", "", body).strip()
    # MetaMathQA bodies carry gsm8k's own "#### N" line: a second answer marker
    # the grader would also read (pitfalls.yaml: double_answer_format).
    body = re.sub(r"^\s*#+\s*[-\d.,$ ]+$", "", body, flags=re.M).strip()
    return f"{body}\n\nANSWER: {answer}{STOP}"


def load_gsm8k(keep_annotations: bool = True):
    from datasets import load_dataset
    d = load_dataset("openai/gsm8k", "main")["train"]
    rows = []
    for i in range(0, len(d) - 500):  # last 500 = private probe pool
        r = d[i]
        reasoning, ans = r["answer"].split("####")
        ans = norm_num(ans)
        if ans is None:
            continue
        reasoning = reasoning.strip()
        if not keep_annotations:
            reasoning = re.sub(r"<<[^>]*>>", "", reasoning)
        rows.append({"question": r["question"], "completion": finalize(reasoning, ans),
                     "gold": ans, "src": "gsm8k"})
    return rows


def load_omi2(limit: int, seed: int = 0):
    from datasets import load_dataset
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M", streaming=True)
    rows = []
    seen = set()
    for r in ds:
        if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
            continue
        ans = norm_num(r["expected_answer"])
        if ans is None:
            continue
        sol = strip_boxed(r["generated_solution"]).strip()
        if len(sol) < 30 or len(sol) > 4000:
            continue
        key = r["problem"].strip()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"question": r["problem"].strip(), "completion": finalize(sol, ans),
                     "gold": ans, "src": r["problem_source"]})
        if len(rows) >= limit:
            break
    return rows


def load_metamath(limit: int):
    from datasets import load_dataset
    ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    rows = []
    seen = set()
    for r in ds:
        if not r["type"].startswith("GSM"):
            continue
        m = re.search(r"The answer is:?\s*([^\n]+)", r["response"])
        if not m:
            continue
        ans = norm_num(m.group(1))
        if ans is None:
            continue
        sol = strip_boxed(r["response"])
        key = r["query"].strip()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"question": key, "completion": finalize(sol, ans),
                     "gold": ans, "src": "metamath"})
        if len(rows) >= limit:
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gsm8k", type=int, default=1, help="1 = include gsm8k train")
    ap.add_argument("--gsm8k-repeat", type=int, default=1)
    ap.add_argument("--omi2", type=int, default=0)
    ap.add_argument("--metamath", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    if args.gsm8k:
        g = load_gsm8k()
        rows += g * args.gsm8k_repeat
        print(f"gsm8k: {len(g)} x{args.gsm8k_repeat}")
    if args.omi2:
        o = load_omi2(args.omi2)
        rows += o
        print(f"omi2: {len(o)}")
    if args.metamath:
        m = load_metamath(args.metamath)
        rows += m
        print(f"metamath: {len(m)}")

    n0 = len(rows)
    rows = [r for r in rows
            if r["completion"].count("ANSWER: ") == 1
            and r["completion"].endswith("ANSWER: " + r["gold"] + STOP)
            and "####" not in r["completion"]]
    print(f"format filter dropped {n0 - len(rows)} rows")

    random.Random(args.seed).shuffle(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")

    # sanity: exactly one answer marker, correct final line
    bad = 0
    for r in rows[:5000]:
        if r["completion"].count("ANSWER: ") != 1 or not r["completion"].endswith(r["gold"] + STOP):
            bad += 1
    print(f"format violations in first 5000: {bad}")


if __name__ == "__main__":
    main()
