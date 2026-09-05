#!/usr/bin/env python3
"""Build GSM8K-style SFT data in the exact format the grader prompts with.

Sources (all train-split derived, no GSM8K test items):
  - nvidia/OpenMathInstruct-2 train_1M, problem_source in {gsm8k, augmented_gsm8k}
  - openai/gsm8k main/train (original reference solutions)

Target format (must match inspect_evals/gsm8k):
  user      : MATH_PROMPT_TEMPLATE.format(prompt=question)
  assistant : <step by step reasoning>\n\nANSWER: <number>
The grader is match(numeric=True, location="end"): the LAST numeric token of the
completion must equal the gold answer, so "ANSWER: N" is the final line and the
marker appears exactly once.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

OMI2_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"
GSM8K_TRAIN = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"

STOP_TOKEN = "<end_of_turn>"

NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def unbox(text: str) -> str:
    """Replace every \\boxed{X} with X (brace-matched)."""
    out = []
    i = 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        k = j + len("\\boxed{")
        depth = 1
        while k < len(text) and depth:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        out.append(text[j + len("\\boxed{"): k - 1])
        i = k


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUM_RE.match(a):
        return None
    if "." in a:
        f = float(a)
        if f == int(f):
            return str(int(f))
        return a.rstrip("0").rstrip(".")
    return str(int(a))


def make_row(question: str, solution: str, answer: str) -> dict | None:
    ans = norm_answer(answer)
    if ans is None:
        return None
    sol = unbox(solution).strip()
    if not sol:
        return None
    # one answer marker only
    if re.search(r"answer\s*:", sol, flags=re.I):
        return None
    if "\\boxed" in sol or "####" in sol:
        return None
    body = f"{sol}\n\nANSWER: {ans}"
    return {
        "prompt": MATH_PROMPT_TEMPLATE.format(prompt=question.strip()),
        # `completion` is the literal model turn INCLUDING the token vLLM stops on,
        # so the file itself is checkable; `completion_body` is the same text bare.
        "completion": body + STOP_TOKEN,
        "completion_body": body,
        "answer": ans,
        "question": question.strip(),
    }


def load_omi2(max_per_problem: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(glob.glob(OMI2_GLOB)):
        t = pq.read_table(f, columns=["problem", "generated_solution", "expected_answer", "problem_source"])
        mask = pc.is_in(t.column("problem_source"), value_set=pa.array(["gsm8k", "augmented_gsm8k"]))
        t = t.filter(mask)
        probs = t.column("problem").to_pylist()
        sols = t.column("generated_solution").to_pylist()
        answs = t.column("expected_answer").to_pylist()
        srcs = t.column("problem_source").to_pylist()
        for p, s, a, src in zip(probs, sols, answs, srcs):
            r = make_row(p, s, a)
            if r is None:
                continue
            r["source"] = f"omi2:{src}"
            by_problem[p].append(r)
        del t
    rows = []
    for p, cands in by_problem.items():
        rng.shuffle(cands)
        seen = set()
        kept = 0
        for c in cands:
            if c["completion_body"] in seen:
                continue
            seen.add(c["completion_body"])
            rows.append(c)
            kept += 1
            if kept >= max_per_problem:
                break
    return rows


def load_gsm8k_train() -> list[dict]:
    f = sorted(glob.glob(GSM8K_TRAIN))[0]
    t = pq.read_table(f)
    rows = []
    for q, a in zip(t.column("question").to_pylist(), t.column("answer").to_pylist()):
        body, _, final = a.rpartition("####")
        body = re.sub(r"<<[^>]*>>", "", body).strip()
        r = make_row(q, body, final)
        if r is None:
            continue
        r["source"] = "gsm8k:train"
        rows.append(r)
    return rows


def fewshot_pool() -> list[str]:
    """Few-shot blocks in exactly inspect_evals' sample_to_fewshot() format."""
    f = sorted(glob.glob(GSM8K_TRAIN))[0]
    t = pq.read_table(f)
    pool = []
    for q, a in zip(t.column("question").to_pylist(), t.column("answer").to_pylist()):
        reasoning, _, final = a.rpartition("####")
        pool.append(f"{q}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {final.strip()}")
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--gsm8k-train-repeat", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=3500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fewshot-frac", type=float, default=0.08,
                    help="fraction of rows whose user turn is prefixed with k solved GSM8K-train examples, "
                         "mirroring the grader's 10-shot system message")
    ap.add_argument("--fewshot-ks", type=str, default="1,2,4")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_omi2(args.max_per_problem, args.seed)
    print(f"omi2 gsm8k-family rows: {len(rows)}")
    g = load_gsm8k_train()
    print(f"gsm8k train rows: {len(g)}")
    rows += g * args.gsm8k_train_repeat

    rows = [r for r in rows if len(r["completion_body"]) <= args.max_chars]
    rng.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    if args.fewshot_frac > 0:
        pool = fewshot_pool()
        ks = [int(k) for k in args.fewshot_ks.split(",")]
        n_fs = int(len(rows) * args.fewshot_frac)
        for r in rows[:n_fs]:
            k = rng.choice(ks)
            shots = rng.sample(pool, k)
            r["prompt"] = "\n\n".join(shots) + "\n\n" + r["prompt"]
            r["fewshot_k"] = k
        rng.shuffle(rows)
        print(f"prefixed {n_fs} rows with k in {ks} solved GSM8K-train examples")

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")
    from collections import Counter
    print(Counter(r["source"] for r in rows))
    lens = sorted(len(r["prompt"]) + len(r["completion"]) for r in rows)
    print("chars p50", lens[len(lens) // 2], "p99", lens[int(len(lens) * 0.99)], "max", lens[-1])


if __name__ == "__main__":
    main()
