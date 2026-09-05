#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt on GSM8K.

Sources (all GSM8K-TRAIN-derived or independent; the official test split is
never read here):
  * nvidia/OpenMathInstruct-2, problem_source in {gsm8k, augmented_gsm8k}
  * openai/gsm8k main train split (original human solutions, cleaned)

Target format, matching the grader (inspect_evals/gsm8k + templates/gemma3.jinja):
  user      : MATH_PROMPT_TEMPLATE.format(prompt=question)
  assistant : <chain of thought>\n\nANSWER: <number>
The chat template appends <end_of_turn> after the assistant turn, so the target
terminator is <end_of_turn>. "ANSWER:" appears exactly once and the last number
in the completion is the answer, which is what match(numeric=True) reads.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re

import pyarrow.parquet as pq

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

OMI_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train-*.parquet"
GSM8K_TRAIN = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
CALC = re.compile(r"<<[^>]*>>")


def unbox(text: str) -> str:
    """Replace \boxed{x} with x and drop $...$ around it; drop LaTeX display math markers."""
    text = BOXED.sub(lambda m: m.group(1), text)
    text = text.replace("\\!", "").replace("\\,", " ")
    return text


def is_clean_integer(ans: str) -> bool:
    a = ans.strip().replace(",", "")
    if a.startswith("-"):
        a = a[1:]
    return a.isdigit() and len(a) <= 12


def normalize_answer(ans: str) -> str:
    return ans.strip().replace(",", "").replace("$", "")


def acceptable_solution(sol: str) -> bool:
    # keep the corpus in plain GSM8K prose: no LaTeX environments, no code
    bad = ["\\begin{", "\\frac", "\\sqrt", "\\times", "\\cdot", "\\left", "\\right",
           "```", "\\[", "\\]", "\\(", "\\)", "\\text{", "$$"]
    return not any(b in sol for b in bad)


def build_target(sol: str, answer: str) -> str | None:
    sol = unbox(sol).strip()
    sol = CALC.sub("", sol)
    if not sol:
        return None
    # the assistant turn ends with the single answer marker the grader reads
    return f"{sol}\n\nANSWER: {answer}"


def load_gsm8k_train() -> list[dict]:
    path = sorted(glob.glob(GSM8K_TRAIN))[0]
    tbl = pq.read_table(path).to_pylist()
    out = []
    for r in tbl:
        q = r["question"].strip()
        body, _, ans = r["answer"].partition("####")
        ans = normalize_answer(ans)
        body = CALC.sub("", body).strip()
        if not is_clean_integer(ans):
            continue
        out.append({"question": q, "solution": body, "answer": ans, "src": "gsm8k_train"})
    return out


def load_omi(max_shards: int) -> list[dict]:
    files = sorted(glob.glob(OMI_GLOB))[:max_shards]
    rows = []
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=20000,
                                     columns=["problem", "generated_solution",
                                              "expected_answer", "problem_source"]):
            for r in batch.to_pylist():
                if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                    continue
                ans = normalize_answer(r["expected_answer"])
                if not is_clean_integer(ans):
                    continue
                sol = r["generated_solution"]
                if not acceptable_solution(sol):
                    continue
                if len(sol) > 2000 or len(r["problem"]) > 1200:
                    continue
                rows.append({"question": r["problem"].strip(), "solution": sol,
                             "answer": ans, "src": r["problem_source"]})
    return rows


def fewshot_block(ex: dict) -> str:
    """Exactly the shape inspect_evals/gsm8k puts in its system message."""
    return f"{ex['question']}\n\nReasoning:\n{ex['solution']}\n\nANSWER: {ex['answer']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--n-omi", type=int, default=45000)
    ap.add_argument("--max-per-problem", type=int, default=1)
    ap.add_argument("--max-shards", type=int, default=4)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--holdout", type=int, default=250, help="dev items held out of GSM8K train")
    ap.add_argument("--dev-out", default="data/dev_train_holdout.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    gsm = load_gsm8k_train()
    rng.shuffle(gsm)
    dev = gsm[: args.holdout]
    gsm = gsm[args.holdout :]
    dev_questions = {d["question"] for d in dev}
    print(f"gsm8k train: {len(gsm)} train / {len(dev)} holdout")

    omi = load_omi(args.max_shards)
    print(f"omi gsm8k-family rows: {len(omi)}")

    # one (or few) solutions per problem, preferring compact ones
    by_q: dict[str, list[dict]] = {}
    for r in omi:
        by_q.setdefault(r["question"], []).append(r)
    picked = []
    for q, rs in by_q.items():
        if q in dev_questions:
            continue
        rs.sort(key=lambda r: len(r["solution"]))
        # middle-length solutions: not the terse one, not the rambling one
        mid = rs[len(rs) // 3 : len(rs) // 3 + args.max_per_problem] or rs[:1]
        picked.extend(mid)
    rng.shuffle(picked)
    picked = picked[: args.n_omi]
    print(f"omi unique problems kept: {len(picked)}")

    pool = picked + gsm
    rng.shuffle(pool)

    # few-shot demo bank: GSM8K train items, in the grader's own demo style
    demo_bank = [g for g in gsm if len(g["solution"]) < 700]

    n_few = int(len(pool) * args.fewshot_frac)
    with open(args.out, "w") as f:
        for i, r in enumerate(pool):
            target = build_target(r["solution"], r["answer"])
            if target is None:
                continue
            user = MATH_PROMPT_TEMPLATE.format(prompt=r["question"])
            messages = []
            if i < n_few:
                k = rng.randint(1, 4)
                demos = rng.sample(demo_bank, k)
                messages.append({"role": "system",
                                 "content": "\n\n".join(fewshot_block(d) for d in demos)})
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": target})
            f.write(json.dumps({"messages": messages, "answer": r["answer"],
                                "src": r["src"],
                                "completion": target + "<end_of_turn>"}) + "\n")

    with open(args.dev_out, "w") as f:
        for i, d in enumerate(dev):
            f.write(json.dumps({"id": f"trdev-{i:04d}", "question": d["question"],
                                "gold": d["answer"]}) + "\n")

    print(f"wrote {args.out} and {args.dev_out} ({len(dev)} dev items)")


if __name__ == "__main__":
    main()
