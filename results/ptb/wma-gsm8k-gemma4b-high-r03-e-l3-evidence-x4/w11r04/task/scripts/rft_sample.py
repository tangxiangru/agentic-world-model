#!/usr/bin/env python3
"""Rejection-sampling: draw k solutions per question from a checkpoint, keep the
ones whose final ANSWER line matches the known answer.

Questions come from GSM8K *train* and from OpenMathInstruct-2's GSM8K-derived
problems (both already in the SFT corpus's provenance).  The grading test split
is never touched.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()
ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]*\.?\d+)")
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def norm(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    if not NUM_RE.match(s):
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


def load_questions(n_gsm: int, n_omi: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    qs = []
    ds = load_dataset("openai/gsm8k", "main", split="train")
    fewshot_qs = set()
    p = os.path.join(TASK_DIR, "data", "eval_fewshot_questions.json")
    if os.path.exists(p):
        fewshot_qs = {q.strip() for q in json.load(open(p))}
    for rec in ds:
        q = rec["question"].strip()
        if q in fewshot_qs:
            continue
        parts = rec["answer"].split("####")
        a = norm(parts[-1])
        if a is not None:
            qs.append({"question": q, "gold": a, "src": "gsm8k_train",
                       "ref": "####".join(parts[:-1]).strip()})
    random.Random(seed).shuffle(qs)
    qs = qs[:n_gsm] if n_gsm else qs

    if n_omi:
        import pyarrow.parquet as pq

        files = sorted(glob.glob(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
            "snapshots/*/data/train_1M-*.parquet"))
        pool, seen = [], set()
        for f in files:
            df = pq.read_table(f).to_pandas()
            df = df[df.problem_source.isin(["gsm8k", "augmented_gsm8k"])]
            for prob, exp, sol in zip(df.problem, df.expected_answer, df.generated_solution):
                prob = str(prob).strip()
                if prob in seen:
                    continue
                seen.add(prob)
                a = norm(str(exp))
                if a is None:
                    continue
                body = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", str(sol))
                if "\\boxed" in body:
                    continue
                pool.append({"question": prob, "gold": a, "src": "omi2",
                             "ref": body.strip()})
        random.Random(seed).shuffle(pool)
        qs += pool[:n_omi]
    return qs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm", type=int, default=0, help="0 = all of gsm8k train")
    ap.add_argument("--n-omi", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-questions", type=int, default=0)
    ap.add_argument("--include-failed-reference", action="store_true",
                    help="for questions no sample solved, emit the dataset's own reference solution")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    with open(os.path.join(TASK_DIR, "templates", "gemma3.jinja")) as f:
        tok.chat_template = f.read()

    qs = load_questions(args.n_gsm, args.n_omi, args.seed)
    if args.limit_questions:
        qs = qs[: args.limit_questions]
    print(f"questions: {len(qs)}", flush=True)

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q["question"])}],
            tokenize=False, add_generation_prompt=True,
        )
        for q in qs
    ]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=2048, dtype="bfloat16", seed=args.seed,
              enable_prefix_caching=True)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate(prompts, sp)

    kept, n_any, n_solved, n_hard_ref = 0, 0, 0, 0
    stats_k = []
    with open(args.out, "w") as f:
        for q, o in zip(qs, outs):
            texts, correct = [], 0
            for c in o.outputs:
                t = c.text
                m = ANS_RE.search(t.strip().splitlines()[-1] if t.strip() else "")
                if m is None:
                    continue
                if norm(m.group(1)) != q["gold"]:
                    continue
                correct += 1
                body = t.strip()
                if body.count("ANSWER:") != 1:
                    continue
                texts.append(body)
            stats_k.append(correct)
            n_any += 1
            if not texts:
                if args.include_failed_reference and q.get("ref"):
                    ref = q["ref"].strip()
                    if "ANSWER:" not in ref and "####" not in ref:
                        row = {"question": q["question"],
                               "completion": f"{ref}\n\nANSWER: {q['gold']}<end_of_turn>",
                               "source": f"hard_ref:{q['src']}", "final": q["gold"]}
                        row["answer"] = row["completion"]
                        f.write(json.dumps(row) + "\n")
                        kept += 1
                        n_hard_ref += 1
                continue
            n_solved += 1
            # prefer shorter, distinct solutions
            uniq, seen = [], set()
            for t in sorted(texts, key=len):
                key = re.sub(r"\s+", " ", t)[:200]
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(t)
            for t in uniq[: args.keep_per_question]:
                row = {"question": q["question"],
                       "completion": t + "<end_of_turn>",
                       "source": f"rft:{q['src']}", "final": q["gold"]}
                row["answer"] = row["completion"]
                f.write(json.dumps(row) + "\n")
                kept += 1
    stats = {"questions": len(qs), "solved_at_least_once": n_solved,
             "pass_rate_any": n_solved / max(1, n_any),
             "mean_correct_per_question": sum(stats_k) / max(1, len(stats_k)) / args.k,
             "rows_written": kept, "hard_reference_rows": n_hard_ref}
    print(json.dumps(stats, indent=2))
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
