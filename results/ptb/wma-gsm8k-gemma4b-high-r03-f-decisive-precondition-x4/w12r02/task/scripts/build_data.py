#!/usr/bin/env python3
"""Build SFT data for GSM8K from nvidia/OpenMathInstruct-2 (GSM8K-derived rows).

Every row is rendered through scripts/render.py, i.e. through the same
templates/gemma3.jinja the grader hands to vLLM, so the training string and
the grading string are produced by one code path.

Output jsonl fields: {"prompt": <str>, "completion": <str>, "answer": <str>,
"source": <str>, "nshot": <int>}
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
import render  # noqa: E402

OMI2_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"
NUM_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{X} with X (brace-matched)."""
    out = []
    i = 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k = j + len("\\boxed{")
        depth = 1
        while k < len(text) and depth:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        out.append(text[j + len("\\boxed{") : k - 1])
        i = k
    return "".join(out)


def norm_answer(a: str) -> str | None:
    a = a.strip()
    if not NUM_RE.match(a):
        return None
    a = a.replace(",", "")
    if a.endswith(".0"):
        a = a[:-2]
    return a


def load_gsm8k_train():
    import datasets

    ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
    return ds


def make_fewshots(gsm_train, rng, k):
    idxs = rng.sample(range(len(gsm_train)), k)
    blocks = []
    for i in idxs:
        r = gsm_train[i]
        parts = r["answer"].split("####")
        target = parts[-1].strip()
        reasoning = "####".join(parts[:-1]).strip()
        blocks.append(render.fewshot_block(r["question"], reasoning, target))
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=140000)
    ap.add_argument("--per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--sources", default="gsm8k,augmented_gsm8k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=0,
                    help="write this many rows to <out>.holdout instead (unique problems)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    keep_sources = set(args.sources.split(","))

    import pyarrow.parquet as pq

    files = sorted(glob.glob(OMI2_GLOB))
    assert files, "OpenMathInstruct-2 parquet shards not found"

    per_problem: dict[str, int] = {}
    rows = []
    n_seen = n_bad_ans = n_bad_sol = 0
    for f in files:
        t = pq.read_table(f)
        cols = {c: t.column(c).to_pylist() for c in
                ("problem", "generated_solution", "expected_answer", "problem_source")}
        for prob, sol, ans, src in zip(cols["problem"], cols["generated_solution"],
                                       cols["expected_answer"], cols["problem_source"]):
            if src not in keep_sources:
                continue
            n_seen += 1
            a = norm_answer(ans)
            if a is None:
                n_bad_ans += 1
                continue
            body = strip_boxed(sol).strip()
            if "ANSWER" in body or "\\boxed" in body:
                n_bad_sol += 1
                continue
            if len(body) < 30 or len(body) > 3500:
                n_bad_sol += 1
                continue
            if per_problem.get(prob, 0) >= args.per_problem:
                continue
            per_problem[prob] = per_problem.get(prob, 0) + 1
            rows.append((prob, body, a, src))
        del t, cols

    print(f"kept {len(rows)} of {n_seen} gsm8k-derived rows "
          f"(bad answer {n_bad_ans}, bad solution {n_bad_sol}), "
          f"{len(per_problem)} unique problems", flush=True)

    rng.shuffle(rows)

    holdout_rows = []
    if args.holdout:
        seen = set()
        rest = []
        for r in rows:
            if len(holdout_rows) < args.holdout and r[3] == "gsm8k" and r[0] not in seen:
                seen.add(r[0])
                holdout_rows.append(r)
            else:
                rest.append(r)
        # drop every other solution of a held-out problem
        rows = [r for r in rest if r[0] not in seen]

    rows = rows[: args.max_rows]

    gsm_train = load_gsm8k_train()

    def emit(fh, rs):
        for prob, body, a, src in rs:
            k = 0
            if rng.random() < args.fewshot_frac:
                k = rng.choice([1, 2, 3, 4, 10])
            fs = make_fewshots(gsm_train, rng, k) if k else None
            prompt = render.build_prompt(prob, fs)
            completion = render.build_completion(body + f"\n\nANSWER: {a}")
            fh.write(json.dumps({"prompt": prompt, "completion": completion,
                                 "answer": a, "source": src, "nshot": k}) + "\n")

    with open(args.out, "w") as fh:
        emit(fh, rows)
    print(f"wrote {len(rows)} -> {args.out}", flush=True)

    if holdout_rows:
        hp = args.out + ".holdout"
        with open(hp, "w") as fh:
            for prob, body, a, src in holdout_rows:
                fh.write(json.dumps({"question": prob, "answer": a, "source": src}) + "\n")
        print(f"wrote {len(holdout_rows)} -> {hp}", flush=True)


if __name__ == "__main__":
    main()
