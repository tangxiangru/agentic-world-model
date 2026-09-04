#!/usr/bin/env python3
"""Build SFT rows for the inspect_evals/gsm8k grader from OpenMathInstruct-2.

Every row is a {prompt, completion} pair where `prompt` is the exact string the
grader's vLLM builds from templates/gemma3.jinja and `completion` is the target
ending in <end_of_turn>.  A configurable share of rows carry a few-shot system
message rendered exactly like inspect_evals builds it, so the model is not
surprised by the 10-shot prefix it always sees at grading time.
"""
from __future__ import annotations

import argparse
import json
import random
import re

from datasets import load_dataset

import render

OMI2 = "nvidia/OpenMathInstruct-2"
BOXED = re.compile(r"\\boxed\s*\{")


def unwrap_boxed(text: str) -> str | None:
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
    out = text
    for _ in range(8):
        m = BOXED.search(out)
        if not m:
            return out
        i = m.end()  # just after '{'
        depth = 1
        while i < len(out) and depth:
            if out[i] == "{":
                depth += 1
            elif out[i] == "}":
                depth -= 1
            i += 1
        if depth:
            return None
        out = out[: m.start()] + out[m.end() : i - 1] + out[i:]
    return None


def clean_solution(sol: str) -> str | None:
    s = unwrap_boxed(sol)
    if s is None:
        return None
    s = s.replace("\\[", "").replace("\\]", "")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def numeric(ans: str) -> bool:
    return bool(re.fullmatch(r"-?\d{1,12}(\.\d{1,6})?", ans.strip()))


def sample_to_fewshot(q: str, reasoning: str, ans: str) -> str:
    """inspect_evals/gsm8k :: sample_to_fewshot, verbatim shape."""
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {ans}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--n-math", type=int, default=0, help="extra augmented_math rows")
    ap.add_argument("--fewshot-share", type=float, default=0.15)
    ap.add_argument("--full-fewshot-share", type=float, default=0.05)
    ap.add_argument("--max-sol-chars", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train_1M", help="OpenMathInstruct-2 split")
    ap.add_argument("--exclude", default=None,
                    help="jsonl already built; its problems are skipped (fresh-rows mode)")
    ap.add_argument("--max-per-problem", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    gsm_train = load_dataset("openai/gsm8k", "main")["train"]
    holdout = set(json.load(open("data/holdout_train_idx.json")))
    demo_pool = []
    holdout_questions = set()
    for i in range(len(gsm_train)):
        r = gsm_train[i]
        if i in holdout:
            holdout_questions.add(r["question"].strip())
            continue
        parts = r["answer"].split("####")
        demo_pool.append(
            (r["question"].strip(), "####".join(parts[:-1]).strip(), parts[-1].strip())
        )
    print(f"[demos] pool={len(demo_pool)} holdout_questions={len(holdout_questions)}")

    full_sys = render.fewshot_system_message()

    excluded = set()
    if args.exclude:
        HEAD = ("Solve the following math problem step by step. The last line of your "
                'response should be of the form "ANSWER: $ANSWER" (without quotes) where '
                "$ANSWER is the answer to the problem.\n\n")
        TAIL = "\n\nRemember to put your answer"
        for line in open(args.exclude):
            pr = json.loads(line)["prompt"]
            i = pr.rindex(HEAD) + len(HEAD)
            excluded.add(pr[i:pr.index(TAIL, i)].strip())
        print(f"[exclude] {len(excluded)} problems from {args.exclude}")

    d = load_dataset(OMI2, split=args.split)
    keep_sources = {"gsm8k", "augmented_gsm8k"}
    if args.n_math:
        keep_sources |= {"math", "augmented_math"}
    d = d.filter(lambda x: x["problem_source"] in keep_sources, num_proc=8)

    by_src = {"gsm": [], "math": []}
    for r in d:
        bucket = "gsm" if r["problem_source"] in ("gsm8k", "augmented_gsm8k") else "math"
        by_src[bucket].append(r)
    print({k: len(v) for k, v in by_src.items()})

    rng.shuffle(by_src["gsm"])
    rng.shuffle(by_src["math"])

    seen_problem = {}
    rows = []
    n_fs = n_full = n_skip_holdout = 0

    def emit(recs, quota):
        nonlocal n_fs, n_full, n_skip_holdout
        made = 0
        for r in recs:
            if made >= quota:
                break
            q = r["problem"].strip()
            if seen_problem.get(q, 0) >= args.max_per_problem:
                continue
            if q in excluded:
                continue
            if q in holdout_questions:
                n_skip_holdout += 1
                continue
            ans = r["expected_answer"].strip()
            if not numeric(ans):
                continue
            sol = r["generated_solution"]
            if len(sol) > args.max_sol_chars:
                continue
            sol = clean_solution(sol)
            if sol is None or not sol:
                continue
            seen_problem[q] = seen_problem.get(q, 0) + 1

            u = rng.random()
            if u < args.full_fewshot_share:
                system = full_sys
                n_full += 1
            elif u < args.full_fewshot_share + args.fewshot_share:
                k = rng.randint(1, 4)
                demos = rng.sample(demo_pool, k)
                system = "\n\n".join(sample_to_fewshot(*dm) for dm in demos)
                n_fs += 1
            else:
                system = None

            completion = f"{sol}\n\nANSWER: {ans}{render.STOP}"
            rows.append(
                {
                    "prompt": render.prompt_for(q, system),
                    "completion": completion,
                    "answer": ans,
                    "source": r["problem_source"],
                }
            )
            made += 1
        return made

    emit(by_src["gsm"], args.n)
    if args.n_math:
        emit(by_src["math"], args.n_math)

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(
        f"[out] {args.out}: {len(rows)} rows; full-10shot={n_full} short-fewshot={n_fs} "
        f"skipped_holdout={n_skip_holdout}"
    )


if __name__ == "__main__":
    main()
