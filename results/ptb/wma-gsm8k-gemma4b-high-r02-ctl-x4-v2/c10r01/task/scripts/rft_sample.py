#!/usr/bin/env python3
"""Rejection-sampling data generation: sample k solutions per problem from a
checkpoint, keep the ones whose final ANSWER matches the reference.

Problems come from OpenMathInstruct-2's GSM8K-derived split and the GSM8K
TRAIN split - never the test split.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

OMI2 = (
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
    "469216e3f46f4dacf476b382e192485ea51a143e/data/train-%05d-of-00032.parquet"
)
GSM8K_TRAIN = (
    "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/"
    "740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet"
)
NUMERIC = re.compile(r"^-?\d{1,12}(\.\d{1,6})?$")
LASTNUM = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)")


def norm(x: str) -> str:
    x = x.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(x)
        if f != f or f in (float("inf"), float("-inf")) or abs(f) > 1e15:
            return x
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        return x


def collect_problems(n_gsm: int, shards: int, seed: int):
    probs, seen = [], set()
    t = pq.read_table(GSM8K_TRAIN).to_pylist()
    for r in t:
        q = r["question"].strip()
        a = r["answer"].split("####")[-1].strip().replace(",", "")
        if NUMERIC.match(a) and q[:200] not in seen:
            seen.add(q[:200])
            probs.append({"question": q, "answer": a, "source": "gsm8k_train"})
    for i in range(shards):
        for r in pq.read_table(OMI2 % i).to_pylist():
            if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                continue
            a = (r["expected_answer"] or "").strip()
            q = r["problem"].strip()
            if not NUMERIC.match(a) or q[:200] in seen:
                continue
            seen.add(q[:200])
            probs.append({"question": q, "answer": a, "source": r["problem_source"]})
            if len(probs) >= n_gsm:
                break
        if len(probs) >= n_gsm:
            break
    random.Random(seed).shuffle(probs)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-problems", type=int, default=40000)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    probs = collect_problems(args.n_problems, args.shards, args.seed)[: args.n_problems]
    print(f"[rft] {len(probs)} problems", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=1536,
        dtype="bfloat16",
        enforce_eager=False,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=[fmt.STOP_TOKEN],
    )
    prompts = [fmt.render_prompt(p["question"]) for p in probs]
    outs = llm.generate(prompts, sp)

    kept, n_any, n_tot = 0, 0, 0
    rng = random.Random(args.seed)
    pool = None
    with open(args.out, "w") as f:
        for p, o in zip(probs, outs):
            cands = []
            try:
                gold = norm(p["answer"])
            except Exception:
                continue
            for c in o.outputs:
                txt = c.text.strip()
                n_tot += 1
                m = re.search(r"ANSWER:\s*\$?(-?[\d,]*\.?\d+)", txt)
                if not m:
                    continue
                try:
                    if norm(m.group(1)) != gold:
                        continue
                except Exception:
                    continue
                # keep only the part up to and including the ANSWER line
                cut = txt[: m.end()].rstrip()
                if len(cut) < 40 or "ANSWER:" != cut[m.start() : m.start() + 7]:
                    pass
                if cut.count("ANSWER:") != 1:
                    continue
                cands.append(cut)
            if not cands:
                continue
            n_any += 1
            uniq, seen_c = [], set()
            for c in cands:
                key = re.sub(r"\s+", " ", c)[:400]
                if key in seen_c:
                    continue
                seen_c.add(key)
                uniq.append(c)
            rng.shuffle(uniq)
            for c in uniq[: args.keep_per_problem]:
                f.write(
                    json.dumps(
                        {
                            "prompt": fmt.render_prompt(p["question"]),
                            "completion": c + fmt.STOP_TOKEN,
                            "question": p["question"],
                            "answer": c,
                            "n_shots": 0,
                            "source": "rft:" + p["source"],
                            "pass_rate": len(cands) / args.k,
                        }
                    )
                    + "\n"
                )
                kept += 1
    print(
        f"[rft] problems with >=1 correct sample: {n_any}/{len(probs)} "
        f"({n_any/len(probs):.3f}); kept {kept} rows from {n_tot} samples -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
