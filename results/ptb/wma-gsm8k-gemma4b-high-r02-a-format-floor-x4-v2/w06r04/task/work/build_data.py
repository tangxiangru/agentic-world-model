"""Build the SFT file: prompts rendered with the grader's own template, targets
that end in the grader's answer marker and the grader's stop token.

Sources (all public HF datasets; no external LLM API is called):
  * openai/gsm8k train rows [0:7273]  - native CoT, the exact eval style.
    Rows [7273:] are the held-out probe set and are never used here.
  * nvidia/OpenMathInstruct-2 train_1M, problem_source in {gsm8k, augmented_gsm8k}
    - Llama-3.1-405B solutions to GSM8K-train problems and to problems augmented
    from GSM8K train, each kept only when it reached the reference answer.

Neither source is derived from the GSM8K test split; both are checked with
../contamination_check.py before use.

Output rows: {id, source, question, prompt, completion}
  prompt     - the full rendered string up to "<start_of_turn>model\\n"
  completion - the assistant turn: reasoning + "\\n\\nANSWER: n" + "<end_of_turn>"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_eval import render, user_message  # noqa: E402

HELD_OUT_FROM = 7273
STOP = "<end_of_turn>"
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
CALC_RE = re.compile(r"<<[^>]*>>")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def norm_num(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    if not NUM_RE.match(s):
        return None
    if "." in s:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    return str(int(s))


def gsm8k_rows():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")["train"]
    out = []
    for i in range(min(HELD_OUT_FROM, len(ds))):
        r = ds[i]
        body, _, tail = r["answer"].partition("####")
        ans = norm_num(tail)
        if ans is None:
            continue
        reasoning = CALC_RE.sub("", body).strip()
        out.append({"id": f"gsm8k-{i}", "source": "openai/gsm8k:train[:7273]",
                    "question": r["question"],
                    "solution": f"{reasoning}\n\nANSWER: {ans}"})
    return out


def fewshot_pool():
    """Few-shot blocks rendered exactly as inspect_evals renders them.

    The grader builds its system message with record_to_sample + sample_to_fewshot
    on RAW gsm8k train rows, so its shots keep the <<3*4=12>> calculator
    annotations. An earlier draft stripped them here, which made every training
    prefix out-of-distribution relative to the one the model meets at grading
    time. These two functions are imported from the installed package, so the
    prefix cannot drift from the harness's.
    """
    from datasets import load_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

    ds = load_dataset("openai/gsm8k", "main")["train"]
    pool = []
    for i in range(min(HELD_OUT_FROM, len(ds))):
        r = ds[i]
        pool.append((r["question"], sample_to_fewshot(record_to_sample(r))))
    return pool


def omi2_rows(cap: int, seed: int):
    import pyarrow.parquet as pq

    files = sorted(glob.glob("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
                             "snapshots/*/data/train_1M-*.parquet"))
    assert files, "OpenMathInstruct-2 train_1M parquet not found"
    rows = []
    for f in files:
        t = pq.read_table(f, columns=["problem", "generated_solution",
                                      "expected_answer", "problem_source"]).to_pylist()
        for r in t:
            if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                continue
            ans = norm_num(r["expected_answer"])
            if ans is None:
                continue
            sol = BOXED_RE.sub(r"\1", r["generated_solution"]).strip()
            if "\\boxed" in sol or len(sol) > 3000:
                continue
            rows.append({"source": f"nvidia/OpenMathInstruct-2:train_1M:{r['problem_source']}",
                         "question": r["problem"].strip(),
                         "solution": f"{sol}\n\nANSWER: {ans}"})
    # dedup on the problem text, keeping the first solution seen
    seen, uniq = set(), []
    for r in rows:
        k = " ".join(r["question"].split()).lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    rng = random.Random(seed)
    rng.shuffle(uniq)
    uniq = uniq[:cap]
    for i, r in enumerate(uniq):
        r["id"] = f"omi2-{i}"
    return uniq


def make_prompt(question: str, shots) -> str:
    msgs = []
    if shots:
        msgs.append({"role": "system", "content": "\n\n".join(block for _, block in shots)})
    msgs.append({"role": "user", "content": user_message(question)})
    return render(msgs, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--omi2-cap", type=int, default=80000)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.30)
    ap.add_argument("--min-shots", type=int, default=3)
    ap.add_argument("--max-shots", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = fewshot_pool()
    base = gsm8k_rows()
    print(f"gsm8k rows: {len(base)}")
    omi = omi2_rows(args.omi2_cap, args.seed)
    print(f"omi2 rows:  {len(omi)}")

    items = base * args.gsm8k_repeat + omi
    rng.shuffle(items)

    n_fs = 0
    with open(args.out, "w") as f:
        for i, r in enumerate(items):
            shots = None
            if rng.random() < args.fewshot_frac:
                k = rng.randint(args.min_shots, args.max_shots)
                shots = [s for s in rng.sample(pool, k + 1) if s[0] != r["question"]][:k]
                n_fs += 1
            f.write(json.dumps({
                "id": f"{i}:{r['id']}",
                "source": r["source"],
                "question": r["question"],
                "prompt": make_prompt(r["question"], shots),
                "completion": r["solution"] + STOP,
            }) + "\n")
    print(f"wrote {len(items)} rows to {args.out} ({n_fs} with a few-shot prefix)")


if __name__ == "__main__":
    main()
