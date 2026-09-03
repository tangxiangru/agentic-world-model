#!/usr/bin/env python3
"""Build SFT data for GSM8K, rendered with the grader's own gemma3 chat template.

Output JSONL rows: {"prompt": <rendered up to <start_of_turn>model\\n>,
                    "completion": <solution + ANSWER line + <end_of_turn>>,
                    "source": <str>, "answer": <str>, "nshot": <int>}

The prompt is produced by the *same* jinja file evaluate.py hands to vLLM
(templates/gemma3.jinja) and the same MATH_PROMPT_TEMPLATE inspect_evals wraps
every question in, so training and grading render byte-identical strings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE

TASK = Path(__file__).resolve().parent
TEMPLATE_PATH = TASK / "templates" / "gemma3.jinja"

STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
CALC_RE = re.compile(r"<<[^>]*>>")
NUM_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def load_template() -> str:
    """Read the grader's own jinja file and record its hash.

    pitfall template_unreachable: training must render with the byte-identical
    template evaluate.py passes to vLLM, not whatever the tokenizer shipped.
    """
    text = TEMPLATE_PATH.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    print("grader template sha256:", sha)
    (TASK / "data" / "template.sha256").parent.mkdir(parents=True, exist_ok=True)
    (TASK / "data" / "template.sha256").write_text(sha + "  templates/gemma3.jinja\n")
    return text


def is_numeric(s: str) -> bool:
    return bool(NUM_RE.match(s.strip()))


def norm_answer(s: str) -> str:
    s = s.strip().replace(",", "")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def clean_solution(sol: str) -> str:
    """Strip \\boxed{} wrappers and gsm8k <<calc>> annotations."""
    sol = CALC_RE.sub("", sol)
    sol = BOXED_RE.sub(r"\1", sol)
    sol = sol.replace("$\\boxed", "").replace("\\boxed", "")
    sol = re.sub(r"[ \t]+\n", "\n", sol)
    sol = re.sub(r"\n{3,}", "\n\n", sol)
    return sol.strip()


def fewshot_block(q: str, a: str) -> str:
    """Exactly inspect_evals.gsm8k.sample_to_fewshot."""
    reasoning, _, target = a.rpartition("####")
    return f"{q}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {target.strip()}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--n-gsm", type=int, default=60000, help="rows from OMI-2 gsm8k sources")
    ap.add_argument("--n-math", type=int, default=8000, help="rows from OMI-2 math sources")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--gsm-train-repeat", type=int, default=1)
    ap.add_argument("--fewshot-rows", type=int, default=4000,
                    help="how many rows carry a few-shot prefix rendered exactly as the "
                         "harness builds its 10-shot system message")
    ap.add_argument("--fewshot-k", type=int, default=10)
    ap.add_argument("--max-sol-chars", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/"
                                       "snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    template = load_template()

    # ---------------- gsm8k train (human CoT) + few-shot pool ----------------
    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    pool = [(r["question"], r["answer"]) for r in gsm]

    records: list[dict] = []

    def add(problem: str, solution: str, answer: str, source: str) -> None:
        records.append({"problem": problem.strip(), "solution": solution,
                        "answer": answer, "source": source})

    for _ in range(args.gsm_train_repeat):
        for q, a in pool:
            reasoning, _, target = a.rpartition("####")
            sol = clean_solution(reasoning)
            add(q, sol, norm_answer(target), "gsm8k_train")

    # ---------------- OpenMathInstruct-2 ----------------
    omi_dir = Path("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
                   "469216e3f46f4dacf476b382e192485ea51a143e/data")
    gsm_rows, math_rows = [], []
    seen_pairs: set[str] = set()
    per_problem: dict[str, int] = {}
    for i in range(args.shards):
        f = omi_dir / f"train-{i:05d}-of-00032.parquet"
        if not f.exists():
            print("missing shard", f)
            continue
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=20000,
                                     columns=["problem", "generated_solution",
                                              "expected_answer", "problem_source"]):
            for r in batch.to_pylist():
                ans = r["expected_answer"]
                if not ans or not is_numeric(ans):
                    continue
                sol_raw = r["generated_solution"]
                if not sol_raw or len(sol_raw) > args.max_sol_chars:
                    continue
                if "\\boxed" not in sol_raw:
                    continue
                prob = r["problem"].strip()
                key = hashlib.md5((prob + "||" + sol_raw).encode()).hexdigest()
                if key in seen_pairs:
                    continue
                pkey = hashlib.md5(prob.encode()).hexdigest()
                if per_problem.get(pkey, 0) >= args.max_per_problem:
                    continue
                sol = clean_solution(sol_raw)
                if not sol or len(sol) < 30:
                    continue
                seen_pairs.add(key)
                per_problem[pkey] = per_problem.get(pkey, 0) + 1
                row = (prob, sol, norm_answer(ans), r["problem_source"])
                if r["problem_source"] in ("gsm8k", "augmented_gsm8k"):
                    gsm_rows.append(row)
                else:
                    math_rows.append(row)
        print(f"shard {i}: gsm={len(gsm_rows)} math={len(math_rows)}", flush=True)
        if len(gsm_rows) >= args.n_gsm and len(math_rows) >= args.n_math:
            break

    rng.shuffle(gsm_rows)
    rng.shuffle(math_rows)
    for prob, sol, ans, src in gsm_rows[: args.n_gsm]:
        add(prob, sol, ans, src)
    for prob, sol, ans, src in math_rows[: args.n_math]:
        add(prob, sol, ans, src)

    rng.shuffle(records)
    print("total records:", len(records))

    # ---------------- render ----------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_bad = 0
    fewshot_idx = set(rng.sample(range(len(records)), min(args.fewshot_rows, len(records))))
    with out.open("w") as fh:
        for i, rec in enumerate(records):
            nshot = 0
            prefix = ""
            if i in fewshot_idx:
                nshot = args.fewshot_k
                shots = rng.sample(pool, nshot)
                prefix = "\n\n".join(fewshot_block(q, a) for q, a in shots) + "\n\n"
            user = prefix + MATH_PROMPT_TEMPLATE.format(prompt=rec["problem"])
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": user}],
                chat_template=template, tokenize=False, add_generation_prompt=True,
            )
            completion = f"{rec['solution']}\n\n{ANSWER_MARKER}{rec['answer']}{STOP_TOKEN}"
            if completion.count(ANSWER_MARKER) != 1 or not completion.endswith(STOP_TOKEN):
                n_bad += 1
                continue
            fh.write(json.dumps({"prompt": prompt, "completion": completion,
                                 "source": rec["source"], "answer": rec["answer"],
                                 "nshot": nshot}) + "\n")
    print("dropped (bad marker/stop):", n_bad)
    print("wrote", out)


if __name__ == "__main__":
    main()
