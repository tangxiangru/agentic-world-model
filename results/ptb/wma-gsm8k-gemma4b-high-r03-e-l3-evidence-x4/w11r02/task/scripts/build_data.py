"""Build the SFT corpus.

Every row is {"question", "solution", "answer", "source"}. `solution` is the
assistant turn body: reasoning, then a single final line "ANSWER: <n>". The
trainer appends <end_of_turn>; nothing here does.

Invariants enforced on every row before it is written (these are the failures
pitfalls.yaml calls double_answer_format and friends):
  * "ANSWER: " occurs exactly once, on the last line,
  * the last number-like token of the solution is the gold answer, because
    inspect's match(numeric=True, location="end") reads exactly that token,
  * no \\boxed, no <<..>> calculator spans, no "The answer is" tail.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

END_OF_TURN = "<end_of_turn>"  # the token vLLM stops on under templates/gemma3.jinja
CALC = re.compile(r"<<[^>]*>>")
BOXED_TAIL = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*\$?\\?boxed\{[^{}]*\}\$?\.?\s*$",
    re.IGNORECASE,
)
BOXED_ANY = re.compile(r"\\boxed\{([^{}]*)\}")
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize_number(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    if v == int(v):
        return str(int(v))
    return None  # gsm8k targets are integers; keep the corpus to that shape


def last_number(text: str) -> str | None:
    words = re.split(r"\s+", text.strip())
    for w in reversed(words):
        w = w.strip(".,:;!?)（）\"'*")
        n = normalize_number(w)
        if n is not None:
            return n
    return None


def finish(reasoning: str, answer: str) -> str | None:
    """Attach the single ANSWER line and validate the row."""
    body = CALC.sub("", reasoning).strip()
    body = BOXED_TAIL.sub("", body).strip()
    body = BOXED_ANY.sub(r"\1", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    if not body or "ANSWER:" in body or "\\boxed" in body:
        return None
    sol = f"{body}\n\nANSWER: {answer}"
    if sol.count("ANSWER: ") != 1:
        return None
    if last_number(sol) != answer:
        return None
    return sol + END_OF_TURN


def load_gsm8k_train():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for r in ds:
        ans = normalize_number(r["answer"].split("####")[-1])
        if ans is None:
            continue
        sol = finish(r["answer"].split("####")[0], ans)
        if sol is None:
            continue
        out.append(
            {"question": r["question"].strip(), "completion": sol, "answer": ans,
             "source": "gsm8k_train"}
        )
    return out


def load_gsm8k_socratic():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "socratic", split="train")
    out = []
    for r in ds:
        ans = normalize_number(r["answer"].split("####")[-1])
        if ans is None:
            continue
        sol = finish(r["answer"].split("####")[0], ans)
        if sol is None:
            continue
        out.append(
            {"question": r["question"].strip(), "completion": sol, "answer": ans,
             "source": "gsm8k_socratic"}
        )
    return out


def load_omi2(max_rows: int, sources=("gsm8k", "augmented_gsm8k"), max_per_problem: int = 2):
    import glob

    import pyarrow.parquet as pq

    files = sorted(
        glob.glob(
            os.path.expanduser(
                "~/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"
            )
        )
    )
    assert files, "OpenMathInstruct-2 train_1M parquet shards not found"
    seen: dict[str, int] = {}
    out = []
    for f in files:
        t = pq.read_table(f, columns=["problem", "generated_solution",
                                      "expected_answer", "problem_source"])
        cols = t.to_pydict()
        for q, s, a, src in zip(cols["problem"], cols["generated_solution"],
                                cols["expected_answer"], cols["problem_source"]):
            if src not in sources:
                continue
            ans = normalize_number(a)
            if ans is None:
                continue
            key = q.strip()
            if seen.get(key, 0) >= max_per_problem:
                continue
            sol = finish(s, ans)
            if sol is None:
                continue
            seen[key] = seen.get(key, 0) + 1
            out.append({"question": key, "completion": sol, "answer": ans,
                        "source": f"omi2_{src}"})
            if len(out) >= max_rows:
                return out
    return out


def load_omi2_split(gsm8k_rows: int, aug_rows: int, per_problem: int):
    a = load_omi2(gsm8k_rows, sources=("gsm8k",), max_per_problem=per_problem)
    b = load_omi2(aug_rows, sources=("augmented_gsm8k",), max_per_problem=1)
    return a + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--omi2-rows", type=int, default=40000)
    ap.add_argument("--omi2-aug-rows", type=int, default=0)
    ap.add_argument("--omi2-per-problem", type=int, default=2)
    ap.add_argument("--sources", default="gsm8k_train,gsm8k_socratic,omi2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    want = set(args.sources.split(","))
    rows = []
    if "gsm8k_train" in want:
        r = load_gsm8k_train()
        print(f"gsm8k_train: {len(r)}", flush=True)
        rows += r
    if "gsm8k_socratic" in want:
        r = load_gsm8k_socratic()
        print(f"gsm8k_socratic: {len(r)}", flush=True)
        rows += r
    if "omi2" in want:
        r = load_omi2_split(args.omi2_rows, args.omi2_aug_rows, args.omi2_per_problem)
        print(f"omi2: {len(r)}", flush=True)
        rows += r

    # exact-duplicate (question, solution) dedup
    seen = set()
    uniq = []
    for r in rows:
        k = (r["question"], r["completion"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    random.Random(args.seed).shuffle(uniq)

    with open(args.out, "w") as f:
        for r in uniq:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(uniq)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
