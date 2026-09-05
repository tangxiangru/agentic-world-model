#!/usr/bin/env python3
"""Build the SFT jsonl for GSM8K.

Sources (both are train-split / augmented-train only, never the gsm8k test split):
  * nvidia/OpenMathInstruct-2, rows with problem_source in {gsm8k, augmented_gsm8k}
  * openai/gsm8k train split, gold solutions

Every target is reformatted so that its last line is "ANSWER: <n>" and it ends
with <end_of_turn>, and every row is verified against a copy of inspect's
match(numeric=True) scorer before it is written.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import END_OF_TURN, grade, sample_to_fewshot, user_prompt  # noqa: E402

OMI_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def clean_omi_solution(sol: str, ans: str) -> str | None:
    # unwrap \boxed{...} -> ...
    body = BOXED_RE.sub(r"\1", sol)
    if "\\boxed" in body or "{" in body and "\\" in body:
        # leftover latex machinery we cannot unwrap safely
        pass
    body = body.replace("$", "").strip()
    if not body:
        return None
    return body + "\n\nANSWER: " + ans


def clean_gsm8k_gold(answer: str) -> tuple[str, str] | None:
    if "####" not in answer:
        return None
    reasoning, target = answer.rsplit("####", 1)
    target = target.strip().replace(",", "")
    if not NUM_RE.match(target):
        return None
    reasoning = reasoning.strip()
    return reasoning, target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--max-omi", type=int, default=60000)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--gsm8k-gold-repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ---------------- few-shot pool (gsm8k train, same construction the grader uses)
    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main")["train"]
    fewshot_pool = []
    gold_rows = []
    for rec in gsm:
        parsed = clean_gsm8k_gold(rec["answer"])
        if parsed is None:
            continue
        reasoning, target = parsed
        fewshot_pool.append((rec["question"], reasoning, target))
        gold_rows.append(
            {
                "question": rec["question"],
                "target_body": reasoning + "\n\nANSWER: " + target,
                "answer": target,
                "src": "gsm8k_train_gold",
            }
        )
    print(f"gsm8k train gold rows: {len(gold_rows)}", flush=True)

    # ---------------- OpenMathInstruct-2
    import pyarrow.parquet as pq

    per_problem: dict[str, int] = {}
    omi_rows = []
    seen_pairs = set()
    for path in sorted(glob.glob(OMI_GLOB)):
        t = pq.read_table(
            path, columns=["problem", "generated_solution", "expected_answer", "problem_source"]
        )
        d = t.to_pydict()
        for prob, sol, ans, src in zip(
            d["problem"], d["generated_solution"], d["expected_answer"], d["problem_source"]
        ):
            if src not in ("gsm8k", "augmented_gsm8k"):
                continue
            ans = (ans or "").strip().replace(",", "")
            if not NUM_RE.match(ans):
                continue
            if per_problem.get(prob, 0) >= args.max_per_problem:
                continue
            body = clean_omi_solution(sol or "", ans)
            if body is None:
                continue
            if len(body) > 4000 or len(prob) > 2000:
                continue
            key = (prob, body)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            per_problem[prob] = per_problem.get(prob, 0) + 1
            omi_rows.append(
                {"question": prob, "target_body": body, "answer": ans, "src": src}
            )
        print(f"  {os.path.basename(path)}: kept {len(omi_rows)}", flush=True)
    print(f"omi rows: {len(omi_rows)} over {len(per_problem)} problems", flush=True)

    rng.shuffle(omi_rows)
    omi_rows = omi_rows[: args.max_omi]

    rows = omi_rows + gold_rows * args.gsm8k_gold_repeat
    rng.shuffle(rows)

    # ---------------- render
    n_bad = 0
    n_fewshot = 0
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            body = r["target_body"].strip()
            if not grade(body, r["answer"]):
                n_bad += 1
                continue
            system = None
            if rng.random() < args.fewshot_frac:
                k = rng.randint(1, 10)
                shots = rng.sample(fewshot_pool, k)
                system = "\n\n".join(sample_to_fewshot(*s) for s in shots)
                n_fewshot += 1
            messages = []
            if system is not None:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user_prompt(r["question"])})
            f.write(
                json.dumps(
                    {
                        "messages": messages,
                        "completion": body + END_OF_TURN,
                        "answer": r["answer"],
                        "src": r["src"],
                    }
                )
                + "\n"
            )
    print(f"wrote {args.out}: {len(rows) - n_bad} rows ({n_bad} dropped, {n_fewshot} few-shot)")


if __name__ == "__main__":
    main()
