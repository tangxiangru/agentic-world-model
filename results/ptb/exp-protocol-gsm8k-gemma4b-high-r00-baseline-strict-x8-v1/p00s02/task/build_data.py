"""Build SFT data for GSM8K from OpenMathInstruct-2 (gsm8k-sourced rows only).

Every row is rendered with the grader's own chat template (templates/gemma3.jinja)
so the training string is byte-identical in shape to what vLLM will see, and the
target ends with the grader's stop token <end_of_turn>.

Answer marker: the final line is exactly "ANSWER: <number>" and the answer is the
last numeric token of the target -- which is what inspect's match(numeric=True,
location="end") reads.

Sources are all derived from the GSM8K TRAIN split (never the test split):
  problem_source in {gsm8k, augmented_gsm8k}.
"""
import argparse
import glob
import json
import os
import random
import re

import pyarrow.parquet as pq

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

OMI2_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"

BOXED_RE = re.compile(r"\\boxed\{")
NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


def strip_boxed(text):
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
    out = []
    i = 0
    while True:
        m = BOXED_RE.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        j = m.end()
        depth = 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        out.append(text[m.end():j - 1])
        i = j
    return "".join(out)


def clean_answer(a):
    a = a.strip().replace(",", "").replace("$", "").replace("\\", "")
    a = a.rstrip(".")
    if a.endswith(".0"):
        a = a[:-2]
    return a


def build_target(solution, answer):
    body = strip_boxed(solution).strip()
    body = body.replace("<<", "").replace(">>", "")
    # drop a dangling final sentence that is nothing but the bare answer, so the
    # ANSWER line is the single place the number is presented as the result
    body = re.sub(r"\n+\s*$", "", body)
    return f"{body}\n\nANSWER: {answer}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--out", default="/home/ben/task/data/sft_omi2_gsm8k.jsonl")
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = sorted(glob.glob(OMI2_GLOB))
    assert files, "OpenMathInstruct-2 shards not found"

    rows = []
    per_problem = {}
    seen = set()
    n_read = 0
    for f in files:
        t = pq.read_table(f)
        srcs = t.column("problem_source").to_pylist()
        probs = t.column("problem").to_pylist()
        sols = t.column("generated_solution").to_pylist()
        answers = t.column("expected_answer").to_pylist()
        for src, p, s, a in zip(srcs, probs, sols, answers):
            n_read += 1
            if src not in ("gsm8k", "augmented_gsm8k"):
                continue
            a = clean_answer(a)
            if not NUMERIC_RE.match(a):
                continue
            if len(s) > args.max_chars or len(p) > 1500:
                continue
            k = per_problem.get(p, 0)
            if k >= args.max_per_problem:
                continue
            tgt = build_target(s, a)
            # the grader reads the LAST numeric token: make sure it is the answer
            toks = tgt.split()
            last_num = next((w for w in reversed(toks)
                             if w.replace(".", "").replace(",", "").isnumeric()), None)
            if last_num is None or last_num.replace(",", "").rstrip(".") != a:
                continue
            if tgt.count("ANSWER:") != 1:
                continue
            h = hash((p, tgt))
            if h in seen:
                continue
            seen.add(h)
            per_problem[p] = k + 1
            rows.append({"problem": p, "target": tgt, "answer": a, "source": src})

    print(f"read {n_read} rows, kept {len(rows)} candidates")
    rng.shuffle(rows)
    rows = rows[: args.n]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps({
                "prompt": MATH_PROMPT_TEMPLATE.format(prompt=r["problem"]),
                "completion": r["target"],
                "answer": r["answer"],
                "source": r["source"],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
