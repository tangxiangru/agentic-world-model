#!/usr/bin/env python3
"""Rejection-sampling data generation: sample solutions from a checkpoint, keep the
ones whose ANSWER line matches gold, write them back out as SFT rows.

Prompts are rendered by scripts/fmt.py, i.e. byte-for-byte the strings the grader
uses, so the samples are drawn from exactly the distribution that gets graded.
Only GSM8K *train* problems (and OpenMathInstruct-2 augmentations of them) are
used as seeds; the test split is never read.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

OMI_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"
ANSWER_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", re.MULTILINE)
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def norm_num(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUM_RE.match(s):
        return None
    return s[:-2] if s.endswith(".0") else s


def first_answer_and_body(text: str):
    """Take the FIRST ANSWER line: anything after it is a continuation past the turn."""
    m = ANSWER_RE.search(text)
    if not m:
        return None, None
    return norm_num(m.group(1)), text[: m.start()].rstrip()


def final_answer(text: str) -> str | None:
    return first_answer_and_body(text)[0]


def load_seeds(n_gsm: int, n_omi: int, seed: int):
    from datasets import load_dataset

    rng = random.Random(seed)
    out = []
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for r in ds:
        a = norm_num(r["answer"].rsplit("####", 1)[-1])
        if a is not None:
            out.append({"question": r["question"].strip(), "gold": a, "src": "gsm8k_train"})
    rng.shuffle(out)
    out = out[:n_gsm]
    if n_omi > 0:
        import pandas as pd

        seen = set()
        aug = []
        for f in sorted(glob.glob(OMI_GLOB)):
            df = pd.read_parquet(f, columns=["problem", "expected_answer", "problem_source"])
            df = df[df.problem_source == "augmented_gsm8k"]
            for p, e in zip(df["problem"], df["expected_answer"]):
                p = p.strip()
                if p in seen:
                    continue
                a = norm_num(e)
                if a is None:
                    continue
                seen.add(p)
                aug.append({"question": p, "gold": a, "src": "omi2_aug"})
        rng.shuffle(aug)
        out += aug[:n_omi]
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-out", default=None)
    ap.add_argument("--n-gsm", type=int, default=7473)
    ap.add_argument("--n-omi", type=int, default=25000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--max-solve-rate", type=float, default=1.01,
                    help="drop problems the model already solves at or above this rate")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seeds = load_seeds(args.n_gsm, args.n_omi, args.seed)
    print(f"[rft] {len(seeds)} seed problems, k={args.k}", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=4096,
              dtype="bfloat16", seed=args.seed)
    # vLLM's offline API stops on the tokenizer's eos (<eos>, id 1) only; the turn
    # terminator the grader stops on is <end_of_turn> (id 106), which lives in
    # generation_config's eos_token_id list and is NOT applied here. Without this the
    # model runs past its answer into a fresh question and every sample is mis-parsed.
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed,
                        stop=[fmt.STOP_TOKEN], stop_token_ids=[1, 106])
    prompts = [fmt.render(s["question"]) for s in seeds]
    outs = llm.generate(prompts, sp)

    kept = defaultdict(list)
    n_correct = n_total = 0
    solve_rate = {}
    raw = open(args.raw_out, "w") if args.raw_out else None
    for s, o in zip(seeds, outs):
        good = []
        for c in o.outputs:
            n_total += 1
            ans, body = first_answer_and_body(c.text.strip())
            if ans is not None and ans == s["gold"] and body:
                n_correct += 1
                good.append(body)
        solve_rate[s["question"]] = len(good) / max(1, len(o.outputs))
        if raw:
            raw.write(json.dumps({"question": s["question"], "gold": s["gold"],
                                  "src": s["src"], "n_correct": len(good),
                                  "n": len(o.outputs)}) + "\n")
        if not good or solve_rate[s["question"]] >= args.max_solve_rate:
            continue
        # prefer distinct, shorter solutions
        good = sorted(set(good), key=len)
        kept[s["question"]] = [(t, s["gold"], s["src"]) for t in good[: args.keep_per_problem]]
    if raw:
        raw.close()

    rng = random.Random(args.seed)
    pool = None
    n_written = 0
    with open(args.out, "w") as f:
        for q, sols in kept.items():
            for body, gold, src in sols:
                if "ANSWER:" in body:
                    continue          # marker must appear exactly once, in the last line
                completion = fmt.target_text(body, gold) + fmt.STOP_TOKEN
                f.write(json.dumps({"prompt": fmt.render(q), "completion": completion,
                                    "src": "rft_" + src, "fewshot": 0}) + "\n")
                n_written += 1
    print(f"[rft] sample pass@1 = {n_correct}/{n_total} = {n_correct/max(1,n_total):.3f}")
    print(f"[rft] {len(kept)} problems contributed, wrote {n_written} rows -> {args.out}")
    _ = pool, rng


if __name__ == "__main__":
    main()
