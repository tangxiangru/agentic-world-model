#!/usr/bin/env python3
"""Build SFT data for GSM8K in the exact format the inspect_evals grader uses.

Rendering follows templates/gemma3.jinja byte-for-byte:

  <bos><start_of_turn>user\n{system + "\n\n" if any}{trim(user)}<end_of_turn>\n<start_of_turn>model\n
  {trim(target)}<end_of_turn>

The user message is inspect_evals.gsm8k.MATH_PROMPT_TEMPLATE; the optional
system part is a k-shot prefix rendered exactly like sample_to_fewshot().

Sources (all GSM8K *train* derived; the official test split is never touched):
  * openai/gsm8k main/train              (7473)
  * nvidia/OpenMathInstruct-2 train_1M, problem_source in {gsm8k, augmented_gsm8k}
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
SOT = "<start_of_turn>"
EOT = "<end_of_turn>"

NUM_RE = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")
CALC_RE = re.compile(r"<<[^>]*>>")
BOXED_RE = re.compile(r"\\boxed\{")


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
    out = []
    i = 0
    while True:
        m = BOXED_RE.search(text, i)
        if m is None:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i : m.start()])
        j = m.end()
        depth = 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        out.append(text[m.end() : j - 1])
        i = j


def render_prompt(question: str, fewshot_prefix: str | None) -> str:
    user = MATH_PROMPT_TEMPLATE.format(prompt=question).strip()
    if fewshot_prefix:
        user = fewshot_prefix + "\n\n" + user
    return f"{BOS}{SOT}user\n{user}{EOT}\n{SOT}model\n"


def render_completion(reasoning: str, answer: str) -> str:
    body = reasoning.strip()
    return f"{body}\n\nANSWER: {answer}{EOT}"


def fewshot_block(question: str, reasoning: str, answer: str) -> str:
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-openmath", type=int, default=60000)
    ap.add_argument("--gsm8k-repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-fewshot-short", type=float, default=0.35)
    ap.add_argument("--p-fewshot-full", type=float, default=0.10)
    ap.add_argument("--eval-fewshot-file", default="data/fewshot_system.txt")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ---- gsm8k train: targets and the pool the few-shot prefixes come from ----
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    gsm_rows = []
    for r in gsm:
        ans_split = r["answer"].split("####")
        target = ans_split[-1].strip()
        reasoning_raw = "####".join(ans_split[:-1]).strip()
        gsm_rows.append(
            {
                "question": r["question"].strip(),
                "reasoning_raw": reasoning_raw,  # keeps <<..>> like the eval prefix
                "reasoning": CALC_RE.sub("", reasoning_raw),
                "answer": target,
            }
        )
    print(f"gsm8k train rows: {len(gsm_rows)}")

    eval_fewshot = Path(args.eval_fewshot_file).read_text()

    # ---- OpenMathInstruct-2, gsm8k-sourced rows only ----
    snap = Path(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
        "469216e3f46f4dacf476b382e192485ea51a143e/data"
    )
    om_rows = []
    seen_problems = set()
    for shard in sorted(snap.glob("train_1M-*.parquet")):
        tbl = pq.read_table(
            shard,
            columns=["problem", "generated_solution", "expected_answer", "problem_source"],
        ).to_pylist()
        for r in tbl:
            if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                continue
            ans = (r["expected_answer"] or "").strip()
            if not NUM_RE.match(ans):
                continue
            q = r["problem"].strip()
            key = q.lower()
            if key in seen_problems:
                continue
            seen_problems.add(key)
            sol = strip_boxed(r["generated_solution"]).strip()
            if not sol or len(sol) > 4000:
                continue
            om_rows.append({"question": q, "reasoning": sol, "answer": ans})
        print(f"  {shard.name}: cumulative {len(om_rows)}")
    rng.shuffle(om_rows)
    om_rows = om_rows[: args.n_openmath]
    print(f"openmath rows kept: {len(om_rows)}")

    # ---- assemble ----
    records = []
    for _ in range(args.gsm8k_repeat):
        for r in gsm_rows:
            records.append(
                {"question": r["question"], "reasoning": r["reasoning"], "answer": r["answer"], "src": "gsm8k_train"}
            )
    for r in om_rows:
        records.append({**r, "src": "openmath2_gsm8k"})
    rng.shuffle(records)

    n_short = n_full = 0
    with open(args.out, "w") as f:
        for rec in records:
            u = rng.random()
            if u < args.p_fewshot_full:
                prefix = eval_fewshot
                n_full += 1
            elif u < args.p_fewshot_full + args.p_fewshot_short:
                k = rng.randint(1, 4)
                shots = rng.sample(gsm_rows, k)
                prefix = "\n\n".join(
                    fewshot_block(s["question"], s["reasoning_raw"], s["answer"]) for s in shots
                )
                n_short += 1
            else:
                prefix = None
            f.write(
                json.dumps(
                    {
                        "prompt": render_prompt(rec["question"], prefix),
                        "completion": render_completion(rec["reasoning"], rec["answer"]),
                        "src": rec["src"],
                    }
                )
                + "\n"
            )
    print(f"wrote {len(records)} rows to {args.out} "
          f"(full-10shot {n_full}, short-kshot {n_short})")


if __name__ == "__main__":
    main()
