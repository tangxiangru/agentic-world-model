"""Rejection-sampling data generation: sample k solutions per problem from a
fine-tuned checkpoint, keep the ones whose 'ANSWER: N' line matches the known
answer, and write them back out in the training format.

Problems come from OpenMathInstruct-2 (which ships an expected_answer); no
GSM8K test item is read.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from build_sft_data import GSM_SOURCES, MATH_SOURCES, load_gsm_train, make_fewshot_block, norm_key  # noqa: E402

OMI2 = sorted(
    glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
    )
)
ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")


def final_answer(text: str) -> str | None:
    m = ANS_RE.findall(text)
    if not m:
        return None
    return common.clean_number(m[-1])


def load_problems(n: int, shard_start: int, math_frac: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out, seen = [], set()
    n_math_max = int(n * math_frac)
    n_math = 0
    for f in OMI2[shard_start:]:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=20_000):
            rows = batch.to_pylist()
            rng.shuffle(rows)
            for r in rows:
                src = r["problem_source"]
                is_gsm = src in GSM_SOURCES
                if not is_gsm and src not in MATH_SOURCES:
                    continue
                if not is_gsm and n_math >= n_math_max:
                    continue
                ans = common.clean_number(r["expected_answer"] or "")
                prob = (r["problem"] or "").strip()
                if ans is None or not prob or len(prob) > 1300:
                    continue
                k = norm_key(prob)
                if k in seen:
                    continue
                seen.add(k)
                if not is_gsm:
                    n_math += 1
                out.append({"question": prob, "answer": ans, "source": src})
                if len(out) >= n:
                    return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-problems", type=int, default=24000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--shard-start", type=int, default=4)
    ap.add_argument("--math-frac", type=float, default=0.2)
    ap.add_argument("--fewshot-frac", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    probs = load_problems(args.n_problems, args.shard_start, args.math_frac, args.seed)
    print(f"[rft] {len(probs)} problems", flush=True)

    gsm_pool = load_gsm_train()
    for i, p in enumerate(probs):
        if i < len(probs) * args.fewshot_frac:
            p["nshot"] = rng.choice([2, 4, 8, 10, 10])
            p["fs"] = make_fewshot_block(gsm_pool, p["nshot"], rng)
        else:
            p["nshot"], p["fs"] = 0, None
        p["prompt"] = common.render_prompt(common.user_prompt(p["question"], p["fs"]))

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=4096,
        dtype="bfloat16",
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    outs = llm.generate([p["prompt"] for p in probs], sp)

    kept, stats = [], defaultdict(int)
    for p, o in zip(probs, outs):
        good, texts = [], set()
        for c in o.outputs:
            t = c.text.strip()
            a = final_answer(t)
            if a is None or a != p["answer"]:
                continue
            if t in texts or "ANSWER:" not in t:
                continue
            texts.add(t)
            good.append(t)
        stats[f"correct_{len(good)}_of_{args.k}"] += 1
        if not good:
            continue
        good.sort(key=len)  # prefer the shorter, cleaner derivations
        for t in good[: args.keep_per_problem]:
            kept.append(
                {
                    "prompt": p["prompt"],
                    "completion": common.render_completion(t),
                    "answer": p["answer"],
                    "source": "rft:" + p["source"],
                    "nshot": p["nshot"],
                    "question": p["question"],
                    "text": p["question"] + "\n\n" + t,
                }
            )

    rng.shuffle(kept)
    with open(args.out, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    print("[rft] stats:", dict(sorted(stats.items())), flush=True)
    print(f"[rft] wrote {len(kept)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
