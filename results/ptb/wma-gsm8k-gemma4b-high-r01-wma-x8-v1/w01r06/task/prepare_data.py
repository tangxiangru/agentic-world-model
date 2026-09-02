#!/usr/bin/env python3
"""Build the SFT set for GSM8K from OpenMathInstruct-2's gsm8k-derived rows.

Everything here is derived from the GSM8K *train* split (directly, or as
NVIDIA's synthetic augmentations of it). The benchmark test split is never
read; the 250 held-out train items in data/dev250.jsonl are excluded so the
private probe stays clean.

Target format is fixed by the grader:
  inspect_evals/gsm8k uses match(numeric=True, location="end"), so the LAST
  number in the completion is the answer, and the prompt template asks for a
  final line 'ANSWER: $ANSWER'. Targets therefore end with 'ANSWER: <int>'
  followed by the grading template's terminator, <end_of_turn>.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

import pyarrow.parquet as pq

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

END_OF_TURN = "<end_of_turn>"
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
INT_RE = re.compile(r"^-?\d+$")


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def clean_solution(sol: str) -> str | None:
    """Strip latex boxing and dollar math so the body reads as plain prose."""
    sol = sol.strip()
    # \boxed{X} -> X  (repeat: some rows nest a boxed inside $...$)
    for _ in range(3):
        new = BOXED.sub(r"\1", sol)
        if new == sol:
            break
        sol = new
    if "\\boxed" in sol:
        return None
    # $...$ around a bare number/short expression -> drop the delimiters
    sol = re.sub(r"\$([^$\n]{1,40})\$", r"\1", sol)
    if "$" in sol.replace("\\$", ""):
        # leftover latex math: skip, it is not gsm8k prose
        pass
    sol = sol.replace("\\%", "%").replace("\\times", "x").replace("\\text{", "").strip()
    if "\\" in sol:
        return None
    return sol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--exclude", default=None,
                    help="jsonl of rows whose problems are already trained on")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--max-rows", type=int, default=200000)
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fewshot-frac", type=float, default=0.08,
                    help="share of rows that carry a k-shot system prefix "
                         "shaped exactly like the harness's own few-shot block")
    args = ap.parse_args()

    dev = {norm_q(json.loads(l)["question"]) for l in open("data/dev250.jsonl")}
    print(f"held-out dev questions: {len(dev)}")

    if args.exclude:
        for line in open(args.exclude):
            r = json.loads(line)
            q = r["prompt"].split("\n\nRemember to put your answer")[0]
            q = q.split("is the answer to the problem.\n\n", 1)[-1]
            dev.add(norm_q(q))
        print(f"held-out + already-trained questions: {len(dev)}")

    files = sorted(
        glob.glob(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
            "snapshots/*/data/train-*.parquet"
        )
    )[args.shard_start: args.shard_start + args.shards]
    print(f"reading {len(files)} shards: {[os.path.basename(f) for f in files]}")

    per_problem: dict[str, int] = {}
    seen_pairs: set[tuple[str, str]] = set()
    rows: list[dict] = []
    stats = {"read": 0, "src": 0, "dev": 0, "notint": 0, "clean": 0, "marker": 0,
             "cap": 0, "dup": 0, "kept": 0}

    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=20000,
                                     columns=["problem", "generated_solution",
                                              "expected_answer", "problem_source"]):
            for r in batch.to_pylist():
                stats["read"] += 1
                if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                    stats["src"] += 1
                    continue
                ans = (r["expected_answer"] or "").strip()
                if not INT_RE.match(ans):
                    stats["notint"] += 1
                    continue
                q = r["problem"].strip()
                nq = norm_q(q)
                if nq in dev:
                    stats["dev"] += 1
                    continue
                if per_problem.get(nq, 0) >= args.max_per_problem:
                    stats["cap"] += 1
                    continue
                sol = clean_solution(r["generated_solution"] or "")
                if sol is None or len(sol) < 20:
                    stats["clean"] += 1
                    continue
                if "answer:" in sol.lower():
                    stats["marker"] += 1
                    continue
                key = (nq, sol[:120])
                if key in seen_pairs:
                    stats["dup"] += 1
                    continue
                seen_pairs.add(key)
                per_problem[nq] = per_problem.get(nq, 0) + 1
                rows.append({
                    "prompt": PROMPT_TEMPLATE.format(prompt=q),
                    # the terminator lives in the data file, not in the
                    # trainer, so what preflight reads is what gets trained
                    "completion": f"{sol}\n\nANSWER: {ans}{END_OF_TURN}",
                    "answer": ans,
                    "source": r["problem_source"],
                    "system": None,
                })
                stats["kept"] += 1
        print(os.path.basename(path), stats)

    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.max_rows]

    # A slice of the training set carries a few-shot system prefix built the
    # same way inspect_evals/gsm8k builds its own (sample_to_fewshot: question,
    # blank line, "Reasoning:", the original train rationale with its <<>>
    # calculator annotations, blank line, "ANSWER: n"; blocks joined by "\n\n").
    # exp-01 showed the base model's failure is non-termination: after emitting
    # "ANSWER: n" it starts a new question, exactly the pattern that block
    # repeats. These rows teach it to stop even with that pattern in context.
    if args.fewshot_frac > 0:
        from datasets import load_dataset
        train = load_dataset("openai/gsm8k", "main")["train"]
        pool = [
            (r["question"].strip(),
             r["answer"].split("####")[0].strip(),
             r["answer"].split("####")[-1].strip())
            for r in train if norm_q(r["question"]) not in dev
        ]
        rng = random.Random(args.seed + 1)
        n_fs = int(len(rows) * args.fewshot_frac)
        for r in rows:
            r["system"] = None
        # k is capped at 3: a 10-shot block is ~1500 tokens, which would push
        # these rows past max_seq_len=1024 and get them dropped, and a longer
        # micro-batch blows the 262k-vocab logits buffer on an 80GB card.
        # k>=1 is enough to teach "stop after ANSWER even with Q/A blocks in
        # context", which is the behaviour exp-01 found missing.
        for r in rows[:n_fs]:
            k = rng.randint(1, 3)
            shots = rng.sample(pool, k)
            r["system"] = "\n\n".join(
                f"{q}\n\nReasoning:\n{reason}\n\nANSWER: {ans}"
                for q, reason, ans in shots
            )
        rng.shuffle(rows)
        print(f"{n_fs} rows carry a 1-10 shot system prefix")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_orig = sum(1 for r in rows if r["source"] == "gsm8k")
    print(f"wrote {len(rows)} rows to {args.out} "
          f"({n_orig} original gsm8k-train problems, {len(rows) - n_orig} augmented)")
    print(f"unique problems: {len(per_problem)}")


if __name__ == "__main__":
    main()
