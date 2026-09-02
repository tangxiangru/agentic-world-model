#!/usr/bin/env python3
"""Sample solutions from a trained checkpoint and keep the ones that reach the
gold answer (rejection-sampling fine-tuning data).

Prompts are rendered zero-shot with scripts/build_data.render_prompt; the
few-shot prefix policy is re-applied when the training file is assembled, so a
kept completion is reusable under any prefix.

Correctness uses the grader's own rule: the LAST number in the completion,
normalised, must equal the gold answer.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_data import NUM_RE, render_prompt  # noqa: E402

NUMBER_TOKEN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text: str) -> float | None:
    m = NUMBER_TOKEN.findall(text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def load_pool(args) -> list[dict]:
    import pyarrow.parquet as pq
    from datasets import load_dataset

    pool = []
    if args.include_gsm8k:
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        for r in gsm:
            pool.append(
                {
                    "question": r["question"].strip(),
                    "answer": r["answer"].split("####")[-1].strip(),
                    "src": "gsm8k_train",
                }
            )
    if args.include_openmath:
        snap = Path(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
            "469216e3f46f4dacf476b382e192485ea51a143e/data"
        )
        seen = set()
        for shard in sorted(snap.glob("train_1M-*.parquet")):
            tbl = pq.read_table(
                shard, columns=["problem", "expected_answer", "problem_source"]
            ).to_pylist()
            for r in tbl:
                if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                    continue
                ans = (r["expected_answer"] or "").strip()
                if not NUM_RE.match(ans):
                    continue
                q = r["problem"].strip()
                k = q.lower()
                if k in seen:
                    continue
                seen.add(k)
                pool.append({"question": q, "answer": ans, "src": "openmath2_gsm8k"})
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-questions", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-gsm8k", type=int, default=1)
    ap.add_argument("--include-openmath", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = load_pool(args)
    rng.shuffle(pool)
    pool = pool[: args.max_questions]
    print(f"[pool] {len(pool)} questions")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        dtype="bfloat16",
        seed=args.seed,
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=None,
    )
    prompts = [render_prompt(p["question"], None) for p in pool]
    outs = llm.generate(prompts, sp)

    n_kept = 0
    n_solved = 0
    with open(args.out, "w") as f:
        for item, out in zip(pool, outs):
            gold = last_number(item["answer"])
            texts = []
            for c in out.outputs:
                t = c.text.strip()
                if not t:
                    continue
                if gold is None or last_number(t) is None:
                    continue
                if abs(last_number(t) - gold) > 1e-6:
                    continue
                texts.append(t)
            if texts:
                n_solved += 1
            # dedup and cap per question
            uniq = []
            seen = set()
            for t in sorted(texts, key=len):
                k = re.sub(r"\s+", " ", t)[:400]
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(t)
            for t in uniq[:2]:
                f.write(
                    json.dumps(
                        {
                            "question": item["question"],
                            "solution": t,
                            "answer": item["answer"],
                            "src": item["src"],
                        }
                    )
                    + "\n"
                )
                n_kept += 1
    print(f"[rft] {n_solved}/{len(pool)} questions solved at least once; {n_kept} rows written to {args.out}")


if __name__ == "__main__":
    main()
