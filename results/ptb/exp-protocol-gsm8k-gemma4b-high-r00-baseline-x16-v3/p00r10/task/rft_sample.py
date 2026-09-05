#!/usr/bin/env python3
"""Rejection-sampling fine-tuning data: sample k solutions per training question from
a checkpoint, keep the ones whose final 'ANSWER: N' matches the reference answer.

Questions come from the GSM8K TRAIN split and from OpenMathInstruct-2's
GSM8K-derived problems (both carry a reference answer). The test split is never read.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import fmt  # noqa: E402

OMI2_GLOB = (
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/"
    "train_*M-*.parquet"
)
ANSWER_RE = re.compile(r"ANSWER:\s*(-?[\d,]+(?:\.\d+)?)")


def norm(a: str) -> str:
    a = a.strip().replace(",", "").replace("$", "")
    if a.endswith(".0"):
        a = a[:-2]
    return a


def gsm8k_train_questions():
    import datasets

    ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
    return [(r["question"].strip(), norm(r["answer"].split("####")[-1])) for r in ds]


def omi2_questions(limit: int, rng: random.Random):
    import pyarrow.parquet as pq

    seen, out = set(), []
    for path in sorted(glob.glob(OMI2_GLOB)):
        df = pq.read_table(path, columns=["problem", "expected_answer", "problem_source"]).to_pandas()
        df = df[df["problem_source"].isin(["gsm8k", "augmented_gsm8k"])]
        for q, a in zip(df["problem"], df["expected_answer"]):
            if not re.fullmatch(r"-?\d+", str(a).strip().replace(",", "")):
                continue
            q = q.strip()
            if q in seen:
                continue
            seen.add(q)
            out.append((q, norm(str(a))))
        if len(out) >= limit * 2:
            break
    rng.shuffle(out)
    return out[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--omi2-questions", type=int, default=0)
    ap.add_argument("--gsm8k-train", type=int, default=1)
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rng = random.Random(args.seed)
    questions = []
    if args.gsm8k_train:
        questions += gsm8k_train_questions()
    if args.omi2_questions:
        questions += omi2_questions(args.omi2_questions, rng)
    print(f"{len(questions)} questions, k={args.k}")

    tok = AutoTokenizer.from_pretrained(fmt.SNAPSHOT)
    prompt_ids = [
        tok(fmt.render_prompt(q, fewshot=False), add_special_tokens=False)["input_ids"]
        for q, _ in questions
    ]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=1536,
        enforce_eager=False,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    outs = llm.generate([{"prompt_token_ids": i} for i in prompt_ids], sampling_params=sp)

    n_kept = n_solved = 0
    with open(args.out, "w") as f:
        for (q, gold), o in zip(questions, outs):
            bodies = []
            for cand in o.outputs:
                t = cand.text.strip()
                m = ANSWER_RE.search(t)
                if not m or norm(m.group(1)) != gold:
                    continue
                body = t[: m.start()].strip()
                if not body or "ANSWER:" in body or len(body) < 20:
                    continue
                if any(body == b for b in bodies):
                    continue
                bodies.append(body)
            if not bodies:
                continue
            n_solved += 1
            # do NOT prefer the shortest: exp-02 showed terse solutions score 0.44 against
            # 0.67 for verbose ones, so shortest-correct would reintroduce the bad style
            rng.shuffle(bodies)
            for body in bodies[: args.max_per_question]:
                fewshot = rng.random() < args.fewshot_frac
                rec = {
                    "prompt": fmt.render_prompt(q, fewshot=fewshot),
                    "completion": fmt.render_target(body, gold),
                    "question": q,
                    "answer": gold,
                    "fewshot": fewshot,
                }
                assert rec["completion"].count(fmt.ANSWER_MARKER) == 1
                f.write(json.dumps(rec) + "\n")
                n_kept += 1
    print(
        f"solved {n_solved}/{len(questions)} questions "
        f"({n_solved / len(questions):.3f}); wrote {n_kept} rows to {args.out}"
    )


if __name__ == "__main__":
    main()
