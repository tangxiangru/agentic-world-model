#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt on GSM8K.

Everything here is derived from the GSM8K *train* split (directly, or via
OpenMathInstruct-2, whose gsm8k / augmented_gsm8k rows are augmentations of the
train split). No test item is read.

The rendered training string is byte-identical to what the grader will send at
eval time, which is why the eval's prompt template and chat template are
reproduced here rather than approximated:

  * MATH_PROMPT_TEMPLATE is copied from inspect_evals/gsm8k/gsm8k.py
  * the chat wrapper is templates/gemma3.jinja, rendered through the same jinja
    the tokenizer uses (see render_chat)
  * the 10-shot system prefix is rebuilt with the same dataset, seed and
    formatter the task uses (fewshot=10, fewshot_seed=42, shuffle=True)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

from datasets import load_dataset  # noqa: E402

# --- copied verbatim from inspect_evals/gsm8k/gsm8k.py -----------------------
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def fewshot_prefix(n: int = 10, seed: int = 42) -> str:
    """Rebuild the grader's 10-shot system message, exactly.

    inspect's hf_dataset(shuffle=True, seed=42, limit=10) shuffles the train
    split with datasets' own shuffle and takes the first 10 rows, then
    sample_to_fewshot renders each one.
    """
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=seed)
    shots = []
    for r in ds.select(range(n)):
        q = r["question"]
        reasoning, target = r["answer"].split("####")
        shots.append(f"{q}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {target.strip()}")
    return "\n\n".join(shots)


# --- target cleaning ---------------------------------------------------------
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
CALC = re.compile(r"<<[^>]*>>")


def clean_solution(sol: str) -> str:
    sol = BOXED.sub(r"\1", sol)
    sol = CALC.sub("", sol)
    sol = sol.replace("$\\", "\\").strip()
    # drop a trailing "#### n" line if the source carried gsm8k's own marker
    sol = re.sub(r"\n*####\s*[-0-9.,]+\s*$", "", sol).strip()
    return sol


NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if not re.fullmatch(r"-?\d+(\.\d+)?", a):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    return a


def build_target(solution: str, answer: str) -> str | None:
    body = clean_solution(solution)
    if "ANSWER:" in body or "####" in body:
        return None
    if not body:
        return None
    return f"{body}\n\nANSWER: {answer}<end_of_turn>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--n-omi", type=int, default=90000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-gsm8k-train", action="store_true", default=True)
    ap.add_argument("--extra", default=None, help="jsonl of already-rendered rows (rft) to merge in")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    prefix = fewshot_prefix()
    print(f"fewshot prefix: {len(prefix)} chars")

    rows: list[dict] = []
    seen_by_problem: dict[str, int] = {}

    # --- source 1: OpenMathInstruct-2, gsm8k + augmented_gsm8k ---------------
    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    omi = omi.filter(lambda r: r["problem_source"] in ("gsm8k", "augmented_gsm8k"), num_proc=8)
    idx = list(range(len(omi)))
    rng.shuffle(idx)
    kept = 0
    for i in idx:
        if kept >= args.n_omi:
            break
        r = omi[i]
        ans = norm_answer(r["expected_answer"])
        if ans is None:
            continue
        key = r["problem"].strip()
        if seen_by_problem.get(key, 0) >= args.max_per_problem:
            continue
        tgt = build_target(r["generated_solution"], ans)
        if tgt is None:
            continue
        seen_by_problem[key] = seen_by_problem.get(key, 0) + 1
        rows.append({"question": key, "target": tgt, "src": r["problem_source"], "answer": ans})
        kept += 1
    print(f"OpenMathInstruct-2 kept: {kept}")

    # --- source 2: the GSM8K train split itself, in the grader's own style ---
    if args.include_gsm8k_train:
        g = load_dataset("openai/gsm8k", "main", split="train")
        n0 = len(rows)
        for r in g:
            reasoning, target = r["answer"].split("####")
            ans = norm_answer(target)
            if ans is None:
                continue
            body = CALC.sub("", reasoning).strip()
            rows.append(
                {
                    "question": r["question"].strip(),
                    "target": f"{body}\n\nANSWER: {ans}<end_of_turn>",
                    "src": "gsm8k_train",
                    "answer": ans,
                }
            )
        print(f"gsm8k train kept: {len(rows) - n0}")

    # --- source 3: already-rendered rows (rejection-sampled self-solutions) --
    extra_rows = []
    if args.extra:
        with open(args.extra) as f:
            extra_rows = [json.loads(l) for l in f]
        print(f"extra rows: {len(extra_rows)}")

    # --- render prompts ------------------------------------------------------
    rng.shuffle(rows)
    out = []
    for k, r in enumerate(rows):
        user = MATH_PROMPT_TEMPLATE.format(prompt=r["question"])
        use_fs = rng.random() < args.fewshot_frac
        system = prefix if use_fs else None
        out.append(
            {
                "id": f"{r['src']}-{k}",
                "system": system,
                "user": user,
                "target": r["target"],
                "answer": r["answer"],
                "src": r["src"],
                "fewshot": bool(use_fs),
            }
        )

    for k, r in enumerate(extra_rows):
        use_fs = rng.random() < args.fewshot_frac
        out.append(
            {
                "id": r.get("id", f"extra-{k}"),
                "system": prefix if use_fs else None,
                "user": r["user"],
                "target": r["target"],
                "answer": r["answer"],
                "src": r.get("src", "extra"),
                "fewshot": bool(use_fs),
            }
        )
    rng.shuffle(out)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} rows to {args.out}")

    # a plain-text dump of question+target only, for the contamination checker
    txt = args.out.replace(".jsonl", "_forcheck.jsonl")
    with open(txt, "w") as f:
        for r in out:
            f.write(json.dumps({"text": r["user"] + "\n" + r["target"]}) + "\n")
    print(f"wrote {txt}")


if __name__ == "__main__":
    main()
