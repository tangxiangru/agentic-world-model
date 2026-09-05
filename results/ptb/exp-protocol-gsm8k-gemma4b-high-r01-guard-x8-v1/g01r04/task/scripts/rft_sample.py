#!/usr/bin/env python3
"""Rejection sampling: draw k solutions per training question from a checkpoint,
keep the ones whose graded answer (the last number, exactly as inspect reads it)
equals the gold answer, and write them back out as SFT rows.

Question pool is GSM8K *train* plus the OpenMathInstruct-2 augmentations of it;
the benchmark test split is never touched.
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

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def graded_number(text: str) -> str | None:
    """What inspect's match(location='end', numeric=True) would extract."""
    m = NUM.findall(text)
    if not m:
        return None
    v = m[-1].replace(",", "").rstrip(".")
    try:
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        return None


def load_pool(n_gsm: int, n_omi: int, dev_path: str, seed: int):
    from datasets import load_dataset

    dev_q = {json.loads(l)["question"] for l in open(dev_path)}
    pool = []
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for r in ds:
        q = r["question"].strip()
        if q in dev_q:
            continue
        a = r["answer"].rpartition("####")[2].strip().replace(",", "")
        pool.append({"question": q, "answer": a, "src": "gsm8k_train"})
    random.Random(seed).shuffle(pool)
    pool = pool[:n_gsm]

    o = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    o = o.filter(lambda x: x["problem_source"] == "augmented_gsm8k", num_proc=8)
    seen = set()
    opool = []
    for r in o:
        q = r["problem"].strip()
        if q in seen or q in dev_q:
            continue
        a = r["expected_answer"].strip().replace(",", "")
        if not re.fullmatch(r"-?\d+", a):
            continue
        seen.add(q)
        opool.append({"question": q, "answer": a, "src": "augmented_gsm8k"})
    random.Random(seed + 1).shuffle(opool)
    return pool + opool[:n_omi]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--n-gsm", type=int, default=7000)
    ap.add_argument("--n-omi", type=int, default=40000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep-per-q", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--dev", default="data/dev_train500.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    args = ap.parse_args()
    print(args, flush=True)

    pool = load_pool(args.n_gsm, args.n_omi, args.dev, args.seed)
    print(f"pool={len(pool)} questions", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=1536,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    sp = SamplingParams(
        # no top_p/top_k: vLLM has no FlashInfer here and falls back to a
        # PyTorch sort over the 262k vocab every step, which dominates runtime
        n=args.k, temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    # the rendered prompt already carries <bos> from the chat template, so feed
    # token ids and never let vLLM add special tokens a second time
    from vllm.inputs import TokensPrompt

    texts = [fmt.render(p["question"], None, system=None) for p in pool]
    ids = fmt.tokenizer()(texts, add_special_tokens=False)["input_ids"]
    outs = llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids], sp)

    # dump raw generations first: a crash in post-processing must never cost
    # the sampling hours again (it did once, on an OverflowError)
    raw_path = args.out + ".raw.jsonl"
    with open(raw_path, "w") as rf:
        for p, o in zip(pool, outs):
            rf.write(json.dumps({"question": p["question"], "answer": p["answer"],
                                 "src": p["src"],
                                 "samples": [c.text for c in o.outputs]}) + "\n")
    print("raw dumped ->", raw_path, flush=True)

    rows, solved, attempted = [], 0, 0
    per_src = {}
    for p, o in zip(pool, outs):
        attempted += 1
        kept = []
        seen_txt = set()
        n_ok = 0
        for c in o.outputs:
            txt = c.text.strip()
            if graded_number(txt) != p["answer"]:
                continue
            n_ok += 1
            key = re.sub(r"\s+", " ", txt)
            if key in seen_txt or len(kept) >= args.keep_per_q:
                continue
            seen_txt.add(key)
            if "ANSWER:" not in txt:
                continue
            kept.append(txt)
        d = per_src.setdefault(p["src"], [0, 0, 0])
        d[0] += 1
        d[1] += n_ok
        d[2] += 1 if n_ok else 0
        if n_ok:
            solved += 1
        for t in kept:
            rows.append({
                "question": p["question"],
                "completion": t + fmt.STOP_TOKEN,
                "src": "rft_" + p["src"],
                "n_correct_of_k": n_ok,
            })

    random.Random(args.seed).shuffle(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats = {
        "questions": attempted,
        "solved_at_least_once": solved,
        "pass_at_k": solved / max(1, attempted),
        "rows": len(rows),
        "per_src": {k: {"q": v[0], "correct_samples": v[1], "solved": v[2],
                        "pass@k": v[2] / max(1, v[0]),
                        "sample_acc": v[1] / max(1, v[0] * args.k)}
                    for k, v in per_src.items()},
    }
    print(json.dumps(stats, indent=2), flush=True)
    if args.stats_out:
        json.dump(stats, open(args.stats_out, "w"), indent=2)
    print(f"wrote {len(rows)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
