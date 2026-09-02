#!/usr/bin/env python3
"""Build the GSM8K SFT corpus.

Every row is {"prompt_question": str, "target_reasoning": str, "answer": str,
              "source": str}.  The trainer renders prompt/target through the
same chat template the grader uses (templates/gemma3.jinja), so this file only
carries the raw pieces.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset, load_from_disk

ROOT = Path(__file__).resolve().parent.parent

CALC_RE = re.compile(r"<<[^>]*>>")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
NUM_RE = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")
INT_RE = re.compile(r"^-?\d+$")
# every one of the 1319 gsm8k test targets is an integer, and a solution that
# argues the question is broken teaches hedging instead of answering
HEDGE_RE = re.compile(
    r"issue with the (question|problem)|problem with the question"
    r"|mistake in the (question|problem)|does not make sense|doesn't make sense"
    r"|cannot be determined|not a whole number|typo",
    re.I,
)


def strip_boxed(text: str) -> str:
    """Remove \\boxed{...} wrappers, keeping their contents.

    The grader's prompt says a boxed command is not needed, and a second answer
    marker is the `double_answer_format` pitfall, so we drop it entirely.
    """
    prev = None
    while prev != text:
        prev = text
        text = BOXED_RE.sub(r"\1", text)
    return text


def clean_solution(text: str) -> str:
    text = strip_boxed(text)
    text = CALC_RE.sub("", text)
    # leftover empty display-math shells such as "\[  \]"
    text = re.sub(r"\\\[\s*\\\]", "", text)
    return text.strip()


def normalise_answer(ans: str) -> str | None:
    """Return the answer as the grader would read it, or None if unusable."""
    a = ans.strip().replace("$", "").replace(",", "").rstrip(".")
    if not NUM_RE.match(a.replace(",", "")):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    return a


def last_number(text: str) -> str | None:
    """The token the grader would extract: last whitespace-token that is numeric."""
    for word in reversed(re.split(r"\s+", text.strip())):
        w = word.strip("$,.:;!?()[]{}%\"'")
        w = w.replace(",", "")
        if w.replace(".", "").replace("-", "").isnumeric():
            return w
    return None


def make_row(question: str, reasoning: str, answer: str, source: str) -> dict | None:
    a = normalise_answer(answer)
    if a is None:
        return None
    if not INT_RE.match(a):
        return None
    reasoning = clean_solution(reasoning)
    if not reasoning or len(reasoning) < 20:
        return None
    if HEDGE_RE.search(reasoning):
        return None
    return {
        "prompt_question": question.strip(),
        "target_reasoning": reasoning,
        "answer": a,
        "source": source,
    }


def build_gsm8k_train() -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for r in ds:
        body, _, ans = r["answer"].rpartition("####")
        row = make_row(r["question"], body, ans, "gsm8k_train")
        if row:
            rows.append(row)
    return rows


def build_omi(max_orig: int, max_aug: int, seed: int) -> list[dict]:
    ds = load_from_disk(str(ROOT / "data" / "omi2_gsm"))
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)

    per_problem: dict[str, int] = {}
    orig, aug = [], []
    for i in idx:
        r = ds[i]
        src = r["problem_source"]
        cap = 2 if src == "gsm8k" else 1
        key = r["problem"].strip()
        if per_problem.get(key, 0) >= cap:
            continue
        if src == "gsm8k" and len(orig) >= max_orig:
            continue
        if src == "augmented_gsm8k" and len(aug) >= max_aug:
            continue
        row = make_row(r["problem"], r["generated_solution"], r["expected_answer"], src)
        if row is None:
            continue
        # the grader reads the last number of the completion; make sure the
        # solution body does not end on a different number than the answer
        per_problem[key] = per_problem.get(key, 0) + 1
        (orig if src == "gsm8k" else aug).append(row)
        if len(orig) >= max_orig and len(aug) >= max_aug:
            break
    return orig + aug


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "sft_v1.jsonl"))
    ap.add_argument("--max-omi-orig", type=int, default=14000)
    ap.add_argument("--max-omi-aug", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = build_gsm8k_train()
    print(f"gsm8k_train: {len(rows)}")
    omi = build_omi(args.max_omi_orig, args.max_omi_aug, args.seed)
    print(f"omi: {len(omi)}")
    rows += omi

    # global dedup on (question, answer)
    seen = set()
    out = []
    for r in rows:
        # keep several distinct solutions to the same problem: they are useful
        # augmentation, so the key includes the solution body
        k = (r["prompt_question"], r["target_reasoning"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)

    rng = random.Random(args.seed)
    rng.shuffle(out)

    with open(args.out, "w") as f:
        for r in out:
            # the trainer appends "\n\nANSWER: {answer}"; store the exact target
            # the literal terminator vLLM stops on (generation_config eos
            # list contains 106 = <end_of_turn>); it is part of the target text
            # so the preflight stop-token check reads the real thing
            r["target"] = f"{r['target_reasoning']}\n\nANSWER: {r['answer']}<end_of_turn>"
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} rows to {args.out}")

    from collections import Counter
    print(Counter(r["source"] for r in out))


if __name__ == "__main__":
    main()
