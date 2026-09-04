#!/usr/bin/env python3
"""Build the SFT corpus for GSM8K-style math word problems.

Sources (all GSM8K *train*-derived or independent; the GSM8K test split is never read):
  - nvidia/OpenMathInstruct-2, train_1M, rows with problem_source in
    {gsm8k, augmented_gsm8k}: Llama-3.1-405B-Instruct solutions to GSM8K train
    problems and to LLM-generated augmentations of them.
  - openai/gsm8k, "main", split=train: the 7473 human-written reference solutions.

Every target is rewritten into the exact shape the grader reads:
    <chain of thought>\n\nANSWER: <number>
and the trainer appends <end_of_turn>, the terminator templates/gemma3.jinja stops on.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import Counter

import pyarrow.parquet as pq

OMI2_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"
KEEP_SOURCES = {"gsm8k", "augmented_gsm8k"}

BOXED_RE = re.compile(r"\\boxed\s*\{")
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
    while True:
        m = BOXED_RE.search(text)
        if m is None:
            return text
        i = m.end()  # just after the '{'
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth:  # unbalanced; give up rather than corrupt the row
            return text
        text = text[: m.start()] + text[m.end() : i - 1] + text[i:]


def normalise_answer(ans: str) -> str | None:
    """Return a plain numeric string, or None if the answer is not a plain number."""
    a = ans.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    a = a.rstrip(".")
    if not NUM_RE.match(a):
        return None
    if "." in a:
        f = float(a)
        if f == int(f):
            a = str(int(f))
        else:
            a = ("%f" % f).rstrip("0").rstrip(".")
    else:
        a = str(int(a))
    return a


def last_number(text: str) -> str | None:
    """Mirror inspect_ai match(numeric=True, location='end'): last numeric token."""
    v = text.strip().replace(",", "").replace("$", "")
    words = re.split(r"\s+", v)
    words.reverse()
    for w in words:
        w2 = w.strip(".!?):;\"'")
        if w2.replace(".", "").replace("-", "").isnumeric():
            return w2
    return None


def make_target(solution: str, answer: str) -> str | None:
    body = strip_boxed(solution).strip()
    if not body:
        return None
    # kill any residual answer marker so 'ANSWER:' appears exactly once
    if "ANSWER:" in body.upper():
        return None
    if "####" in body:
        body = body.split("####")[0].rstrip()
    return f"{body}\n\nANSWER: {answer}"


def load_omi2() -> list[dict]:
    files = sorted(glob.glob(OMI2_GLOB))
    assert files, "OpenMathInstruct-2 parquet files not found"
    out = []
    for path in files:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(
                rg, columns=["problem", "generated_solution", "expected_answer", "problem_source"]
            )
            for r in tbl.to_pylist():
                if r["problem_source"] not in KEEP_SOURCES:
                    continue
                out.append(r)
    return out


def load_gsm8k_train() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for r in ds:
        body, _, ans = r["answer"].rpartition("####")
        out.append(
            {
                "problem": r["question"],
                "generated_solution": body.strip(),
                "expected_answer": ans.strip(),
                "problem_source": "gsm8k_human",
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/sft_pool.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_omi2() + load_gsm8k_train()
    print(f"raw rows: {len(rows)}  {Counter(r['problem_source'] for r in rows)}")

    per_problem: dict[str, list[dict]] = {}
    stats = Counter()
    for r in rows:
        ans = normalise_answer(r["expected_answer"])
        if ans is None:
            stats["drop_nonnumeric_answer"] += 1
            continue
        tgt = make_target(r["generated_solution"], ans)
        if tgt is None:
            stats["drop_bad_target"] += 1
            continue
        if last_number(tgt) != last_number("x " + ans):
            stats["drop_last_number_mismatch"] += 1
            continue
        n_words = len(tgt.split())
        if n_words < 12 or n_words > 600:
            stats["drop_length"] += 1
            continue
        per_problem.setdefault(r["problem"].strip(), []).append(
            {"problem": r["problem"].strip(), "target": tgt, "answer": ans,
             "source": r["problem_source"], "n_words": n_words}
        )
        stats["kept"] += 1

    print(f"filter stats: {dict(stats)}  unique problems: {len(per_problem)}")

    out_rows = []
    for prob, cands in per_problem.items():
        # dedup identical targets, prefer the shortest distinct solutions (less rambling)
        seen = set()
        uniq = []
        for c in sorted(cands, key=lambda c: c["n_words"]):
            key = c["target"]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        # human gsm8k reference solutions are always kept
        human = [c for c in uniq if c["source"] == "gsm8k_human"]
        rest = [c for c in uniq if c["source"] != "gsm8k_human"]
        take = human + rest[: max(0, args.max_per_problem - len(human))]
        out_rows.extend(take)

    rng.shuffle(out_rows)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out_rows)} rows -> {args.out}")
    print(f"by source: {Counter(r['source'] for r in out_rows)}")


if __name__ == "__main__":
    main()
