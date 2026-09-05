"""Build the SFT corpus in the exact format the grader reads.

Every row is {"question", "solution", "answer", "src"}; the solution always
ends with a single "ANSWER: <number>" line, which is the last numeric token in
the text (the grader's match(numeric=True, location="end") reads exactly that).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

import pyarrow.parquet as pq
from datasets import load_dataset

from prompt_fmt import END_OF_TURN

HERE = os.path.dirname(os.path.abspath(__file__))
OMI2_DIR = (
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
    "469216e3f46f4dacf476b382e192485ea51a143e/data"
)
NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").replace("%", "")
    if not NUM_RE.match(a):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    return a


def finish(body: str, answer: str) -> str:
    body = body.strip()
    # a de-boxed solution often ends with a bare "45" line; the ANSWER line
    # replaces it rather than sitting after it
    lines = body.split("\n")
    while lines and NUM_RE.match(lines[-1].strip().replace(",", "")):
        lines.pop()
    body = "\n".join(lines).strip()
    return f"{body}\n\nANSWER: {answer}"


def gsm8k_rows(indices: list[int]) -> list[dict]:
    train = load_dataset("openai/gsm8k", "main")["train"]
    out = []
    for i in indices:
        q, a = train[i]["question"], train[i]["answer"]
        body, gold = a.split("####")
        gold = norm_answer(gold)
        if gold is None:
            continue
        out.append(
            {"question": q.strip(), "completion": finish(body, gold) + END_OF_TURN,
             "answer": gold, "src": "gsm8k_train"}
        )
    return out


def omi2_rows(shards: int, per_problem: int, max_rows: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    files = sorted(
        os.path.join(OMI2_DIR, f) for f in os.listdir(OMI2_DIR) if f.endswith(".parquet")
    )[:shards]
    seen: dict[str, int] = {}
    out = []
    bad_latex = ("\\[", "\\frac", "\\sqrt", "\\begin{", "\\text{", "$$")
    for path in files:
        t = pq.read_table(path, columns=["problem", "generated_solution",
                                         "expected_answer", "problem_source"])
        cols = t.to_pydict()
        for q, sol, ans, src in zip(cols["problem"], cols["generated_solution"],
                                    cols["expected_answer"], cols["problem_source"]):
            if src not in ("gsm8k", "augmented_gsm8k"):
                continue
            gold = norm_answer(ans)
            if gold is None:
                continue
            if len(sol) > 1800 or len(q) > 1200:
                continue
            if any(b in sol for b in bad_latex):
                continue
            n = seen.get(q, 0)
            if n >= per_problem:
                continue
            body = BOXED_RE.sub(r"\1", sol)
            body = body.replace("$", "")
            # the appended ANSWER line must be the only trailing marker
            if "ANSWER:" in body:
                continue
            seen[q] = n + 1
            out.append({"question": q.strip(),
                        "completion": finish(body, gold) + END_OF_TURN,
                        "answer": gold, "src": src})
        if len(out) >= max_rows * 3:
            break
    rng.shuffle(out)
    return out[:max_rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--omi2-shards", type=int, default=8)
    ap.add_argument("--omi2-per-problem", type=int, default=2)
    ap.add_argument("--omi2-max", type=int, default=60000)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    split = json.load(open(os.path.join(HERE, "data/split_idx.json")))
    native = gsm8k_rows(split["train"])
    print(f"gsm8k train pool rows: {len(native)}")
    omi = omi2_rows(args.omi2_shards, args.omi2_per_problem, args.omi2_max, args.seed)
    print(f"omi2 rows: {len(omi)}")

    # hold the probe questions out of every pool, including paraphrases keyed on
    # exact text (augmented items are new problems, so this only removes exact hits)
    probe_qs = {json.loads(l)["question"] for l in
                open(os.path.join(HERE, "analysis/probe250.jsonl"))}
    rows = native * args.gsm8k_repeat + omi
    rows = [r for r in rows if r["question"] not in probe_qs]

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    with open(os.path.join(HERE, args.out), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
