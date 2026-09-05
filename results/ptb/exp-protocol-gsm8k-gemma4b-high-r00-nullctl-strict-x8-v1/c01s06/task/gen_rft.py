#!/usr/bin/env python3
"""Rejection sampling: draw k solutions per training problem from the current
policy, keep the ones that reach the reference answer, and write a new SFT file."""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq
from datasets import load_dataset

from prep_data import MATH_PROMPT_TEMPLATE, OMI_GLOB, clean_answer

NUMWORD = re.compile(r"^-?\d+(\.\d+)?$")


def last_number(text: str) -> str | None:
    """Mirror inspect's match(location='end', numeric=True) extraction."""
    v = text.strip().replace(",", "").replace("$", "")
    words = re.split(r"\s+", v)
    for w in reversed(words):
        w = w.strip().rstrip(".").rstrip("%").strip("*").strip()
        if w.replace(".", "").replace("-", "").isnumeric() or NUMWORD.match(w):
            try:
                return format(float(w), ".5g")
            except ValueError:
                return None
    return None


def norm(a: str) -> str | None:
    try:
        return format(float(a), ".5g")
    except ValueError:
        return None


def build_prompts(n_omi: int, seed: int):
    items = []
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for r in ds:
        ans = clean_answer(r["answer"].split("####")[1])
        if ans:
            items.append({"q": r["question"].strip(), "a": ans, "src": "gsm8k_train"})
    seen = {it["q"] for it in items}
    omi = []
    for f in sorted(glob.glob(OMI_GLOB)):
        tbl = pq.read_table(f, columns=["problem", "expected_answer", "problem_source"])
        for p, a, s in zip(tbl.column("problem").to_pylist(),
                           tbl.column("expected_answer").to_pylist(),
                           tbl.column("problem_source").to_pylist()):
            if s != "augmented_gsm8k":
                continue
            p = p.strip()
            if p in seen or len(p) > 1200:
                continue
            ans = clean_answer(a)
            if not ans:
                continue
            seen.add(p)
            omi.append({"q": p, "a": ans, "src": "omi_aug"})
    rng = random.Random(seed)
    rng.shuffle(omi)
    items.extend(omi[:n_omi])
    rng.shuffle(items)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckpt/sft1")
    ap.add_argument("--out", default="data/rft_raw.jsonl")
    ap.add_argument("--n-omi", type=int, default=20000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--k-gsm", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    items = build_prompts(args.n_omi, args.seed)
    print(f"{len(items)} problems")

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              max_model_len=2048, dtype="bfloat16", enforce_eager=False,
              disable_log_stats=True)

    def wrap(q):
        body = MATH_PROMPT_TEMPLATE.format(prompt=q)
        return f"<bos><start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n"

    groups = defaultdict(list)
    for grp_k in {args.k, args.k_gsm}:
        idxs = [i for i, it in enumerate(items)
                if (args.k_gsm if it["src"] == "gsm8k_train" else args.k) == grp_k]
        if not idxs:
            continue
        prompts = [wrap(items[i]["q"]) for i in idxs]
        sp = SamplingParams(n=grp_k, temperature=args.temp, top_p=0.95, top_k=64,
                            max_tokens=args.max_tokens, seed=None,
                            stop=["<end_of_turn>", "<start_of_turn>"],
                            stop_token_ids=[1, 106])
        outs = llm.generate(prompts, sp)
        for i, o in zip(idxs, outs):
            groups[i] = [c.text for c in o.outputs]

    n_kept = 0
    with open(args.out, "w") as f:
        for i, texts in groups.items():
            it = items[i]
            tgt = norm(it["a"])
            correct = []
            for t in texts:
                t = t.strip()
                if not t or "ANSWER:" not in t:
                    continue
                if last_number(t) == tgt:
                    correct.append(t)
            f.write(json.dumps({"question": it["q"], "answer": it["a"], "src": it["src"],
                                "n": len(texts), "n_correct": len(correct),
                                "correct": correct}) + "\n")
            n_kept += len(correct)
    print(f"wrote {args.out}: {n_kept} correct samples over {len(groups)} problems")


if __name__ == "__main__":
    main()
