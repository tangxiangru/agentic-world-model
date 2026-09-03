#!/usr/bin/env python3
"""Build the SFT corpus for GSM8K, formatted exactly the way the grader reads it.

The grader (inspect_evals/gsm8k) wraps every test question in MATH_PROMPT_TEMPLATE,
prepends a 10-shot system message built from the gsm8k TRAIN split, renders the
conversation with templates/gemma3.jinja and scores match(numeric=True) on the tail
of the completion. So every training target here ends with:

    ANSWER: <number><end_of_turn>

Sources (none touches the gsm8k test split):
  * openai/gsm8k main/train      -- 7473 reference chains
  * nvidia/OpenMathInstruct-2    -- solutions whose problem_source contains "gsm8k"
                                    (built from the gsm8k train split by NVIDIA)
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path("/home/ben/task")
RAW = ROOT / "data" / "raw"

# Verbatim from inspect_evals/gsm8k/gsm8k.py (MATH_PROMPT_TEMPLATE, .strip()ed there).
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

END_OF_TURN = "<end_of_turn>"
CALC = re.compile(r"<<[^>]*>>")
BOXED = re.compile(r"\\boxed\s*\{")
NUMLIKE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def clean_number(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUMLIKE.match(s):
        return None
    # drop a trailing ".0" so the target matches gsm8k's integer gold format
    if s.endswith(".0"):
        s = s[:-2]
    return s


def strip_boxed_tail(text: str) -> str:
    """Remove the trailing sentence/line that carries the \\boxed{...} answer."""
    m = None
    for m in BOXED.finditer(text):
        pass
    if m is None:
        return text.rstrip()
    cut = text.rfind("\n", 0, m.start())
    head = text[: cut if cut != -1 else m.start()]
    # also drop a dangling "The final answer is" fragment if it sat on the same line
    head = re.sub(r"(?:\n|^)[^\n]*(final answer|answer is)[^\n]*$", "", head, flags=re.I)
    return head.rstrip()


def build_user(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question.strip())


def gsm8k_rows(path: Path):
    """gsm8k train reference chains: drop calculator annotations, retarget the marker."""
    out = []
    for line in path.open():
        r = json.loads(line)
        q = r["question"].strip()
        ans = r["answer"]
        body, _, final = ans.rpartition("####")
        gold = clean_number(final)
        if gold is None:
            continue
        body = CALC.sub("", body).strip()
        if not body:
            continue
        out.append({"question": q, "solution": f"{body}\n\nANSWER: {gold}", "gold": gold,
                    "src": "gsm8k_train"})
    return out


def omi2_rows(path: Path, per_problem: int, rng: random.Random):
    by_problem: dict[str, list[dict]] = {}
    for line in path.open():
        r = json.loads(line)
        gold = clean_number(str(r["expected_answer"]))
        if gold is None:
            continue
        sol = strip_boxed_tail(r["generated_solution"])
        if not sol or "\\boxed" in sol or "####" in sol:
            continue
        # a \boxed on the very first line leaves a stump, not a chain of thought
        if len(sol) < 60 or len(sol) > 3500:
            continue
        by_problem.setdefault(r["problem"].strip(), []).append(
            {"question": r["problem"].strip(),
             "solution": f"{sol}\n\nANSWER: {gold}",
             "gold": gold,
             "src": r["problem_source"]}
        )
    out = []
    for _, cands in by_problem.items():
        seen = set()
        uniq = []
        for c in cands:
            k = c["solution"]
            if k in seen:
                continue
            seen.add(k)
            uniq.append(c)
        rng.shuffle(uniq)
        out.extend(uniq[:per_problem])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-problem", type=int, default=2)
    ap.add_argument("--max-omi2", type=int, default=45000)
    ap.add_argument("--gsm8k-repeat", type=int, default=1)
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "sft_v1.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", type=str, default=str(ROOT / "data" / "probe250.jsonl"),
                    help="jsonl of held-out {question}: those problems never enter training")
    args = ap.parse_args()

    held = set()
    if args.exclude and Path(args.exclude).exists():
        held = {json.loads(l)["question"].strip() for l in Path(args.exclude).open()}
        print(f"held-out probe questions: {len(held)}")

    rng = random.Random(args.seed)
    g = [r for r in gsm8k_rows(RAW / "gsm8k_train.jsonl") if r["question"] not in held]
    print(f"gsm8k_train rows (probe removed): {len(g)}")
    o = [r for r in omi2_rows(RAW / "omi2_gsm8k.jsonl", args.per_problem, rng)
         if r["question"] not in held]
    print(f"omi2 rows (<= {args.per_problem}/problem, probe removed): {len(o)}")
    rng.shuffle(o)
    o = o[: args.max_omi2]

    rows = g * args.gsm8k_repeat + o
    rng.shuffle(rows)

    outp = Path(args.out)
    with outp.open("w") as f:
        for r in rows:
            f.write(json.dumps({
                "question": r["question"],
                "prompt": build_user(r["question"]),
                # the stop token is part of the stored target so the preflight
                # stop_token_consistent check verifies the real training string
                "completion": r["solution"] + END_OF_TURN,
                "gold": r["gold"],
                "src": r["src"],
            }) + "\n")
    print(f"wrote {len(rows)} rows -> {outp}")

    from collections import Counter
    print(Counter(r["src"] for r in rows))


if __name__ == "__main__":
    main()
