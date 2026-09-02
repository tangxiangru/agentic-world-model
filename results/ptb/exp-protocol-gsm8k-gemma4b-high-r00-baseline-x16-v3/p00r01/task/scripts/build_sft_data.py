#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt on GSM8K.

Sources (both derived from the GSM8K *train* split only - never the test split):
  * nvidia/OpenMathInstruct-2, problem_source in {gsm8k, augmented_gsm8k}
  * openai/gsm8k, split=train

Every target is reshaped so that the LAST numeric token of the completion is the
answer the grader reads: the body, then a blank line, then "ANSWER: <n>".
`\\boxed{}` and gsm8k's own "#### n" marker are removed so the target carries the
answer marker exactly once (pitfalls.yaml double_answer_format).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

from datasets import load_dataset, load_from_disk

ANSWER_MARKER = "ANSWER: "
# the token vllm stops on: generation_config.eos_token_id == [1, 106], 106 is
# <end_of_turn>, which is also what templates/gemma3.jinja closes a turn with.
END_OF_TURN = "<end_of_turn>"
MAX_SOLUTION_CHARS = 3000
MIN_SOLUTION_CHARS = 40


def find_boxed(text: str) -> list[str]:
    """Return the contents of every \\boxed{...} in text (brace-balanced)."""
    out = []
    for m in re.finditer(r"\\boxed\{", text):
        i = m.end()
        depth = 1
        start = i
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            out.append((m.start(), i, text[start : i - 1]))
    return out


def normalize_int(s: str) -> str | None:
    """Return a canonical integer string, or None if s is not a plain integer."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    s = s.replace("\\!", "").replace("\\,", "").replace("{", "").replace("}", "")
    if re.fullmatch(r"-?\d+", s):
        return str(int(s))
    if re.fullmatch(r"-?\d+\.0+", s):
        return str(int(float(s)))
    return None


def clean_omi2(solution: str, answer: str) -> str | None:
    boxes = find_boxed(solution)
    if len(boxes) != 1:
        return None
    start, end, inner = boxes[0]
    if normalize_int(inner) != answer:
        return None
    # drop the boxed markup, keep the number in place
    body = solution[:start] + inner + solution[end:]
    return body


def strip_calc_annotations(text: str) -> str:
    return re.sub(r"<<[^>]*>>", "", text)


def finalize(body: str, answer: str) -> str | None:
    body = body.strip()
    if not (MIN_SOLUTION_CHARS <= len(body) <= MAX_SOLUTION_CHARS):
        return None
    if "ANSWER:" in body or "\\boxed" in body or "####" in body:
        return None
    return f"{body}\n\n{ANSWER_MARKER}{answer}{END_OF_TURN}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/sft_v1.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--max-omi2", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    rows: list[dict] = []

    # ---- 1. OpenMathInstruct-2 (gsm8k + augmented_gsm8k) --------------------
    omi2 = load_from_disk("/home/ben/task/data/omi2_gsm8k")
    per_problem: dict[str, list[str]] = defaultdict(list)
    kept = dropped_box = dropped_ans = dropped_len = 0
    for r in omi2:
        ans = normalize_int(r["expected_answer"])
        if ans is None:
            dropped_ans += 1
            continue
        body = clean_omi2(r["generated_solution"], ans)
        if body is None:
            dropped_box += 1
            continue
        tgt = finalize(body, ans)
        if tgt is None:
            dropped_len += 1
            continue
        per_problem[r["problem"]].append(tgt)
        kept += 1
    print(f"omi2: kept={kept} drop_box={dropped_box} drop_ans={dropped_ans} "
          f"drop_len={dropped_len} unique_problems={len(per_problem)}")

    omi2_rows = []
    for q, tgts in per_problem.items():
        # dedup identical solutions, then keep at most max-per-problem
        seen, uniq = set(), []
        for t in tgts:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        rng.shuffle(uniq)
        for t in uniq[: args.max_per_problem]:
            omi2_rows.append({"question": q, "target": t, "source": "omi2_gsm8k"})
    rng.shuffle(omi2_rows)
    omi2_rows = omi2_rows[: args.max_omi2]
    rows.extend(omi2_rows)
    print(f"omi2 rows after per-problem cap and sampling: {len(omi2_rows)}")

    # ---- 2. openai/gsm8k train (native terse style) ------------------------
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    n_gsm = 0
    for r in gsm:
        body, _, tail = r["answer"].rpartition("####")
        ans = normalize_int(tail)
        if ans is None:
            continue
        body = strip_calc_annotations(body).strip()
        tgt = finalize(body, ans)
        if tgt is None:
            continue
        rows.append({"question": r["question"], "target": tgt, "source": "gsm8k_train"})
        n_gsm += 1
    print(f"gsm8k_train rows: {n_gsm}")

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
