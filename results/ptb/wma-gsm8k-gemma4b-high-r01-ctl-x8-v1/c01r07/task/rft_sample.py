#!/usr/bin/env python3
"""Rejection-sampling data generation from a fine-tuned checkpoint.

Samples k solutions per GSM8K *train* question with vLLM under the grader's own
prompt, keeps the ones whose final number matches the gold answer, and writes an
SFT jsonl in the same prompt/completion shape as build_data.py.

No benchmark test item is read: questions come from openai/gsm8k train only, and
the 300 probe questions are excluded.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re

import pyarrow.dataset as pds

import prompt_spec as ps

GSM_GLOB = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"
OMI_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def qkey(q: str) -> str:
    return hashlib.md5(re.sub(r"\W+", " ", q.lower()).strip().encode()).hexdigest()


def last_number(text: str) -> str | None:
    """What match(numeric=True, location='end') will read off the completion."""
    toks = re.split(r"\s+", text.strip())
    for w in reversed(toks):
        w2 = w.strip().replace(",", "").replace("$", "").rstrip(".:)%")
        if NUM.fullmatch(w2):
            return w2.rstrip("0").rstrip(".") if "." in w2 else w2
    return None


def norm(x: str) -> str:
    x = x.replace(",", "").replace("$", "").strip()
    return x.rstrip("0").rstrip(".") if "." in x else x


def load_questions(source: str, held: set):
    if source == "gsm8k_train":
        rows = pds.dataset(sorted(glob.glob(GSM_GLOB))).to_table().to_pylist()
        out = [{"question": r["question"], "gold": norm(r["answer"].split("####")[-1])}
               for r in rows]
    else:  # augmented_gsm8k problems from OpenMathInstruct-2
        tbl = pds.dataset(sorted(glob.glob(OMI_GLOB))).to_table(
            filter=pds.field("problem_source").isin(["augmented_gsm8k", "gsm8k"]),
            columns=["problem", "expected_answer"])
        seen, out = set(), []
        for r in tbl.to_pylist():
            k = qkey(r["problem"])
            if k in seen:
                continue
            seen.add(k)
            g = norm(r["expected_answer"])
            if NUM.fullmatch(g):
                out.append({"question": r["problem"], "gold": g})
    return [r for r in out if qkey(r["question"]) not in held]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--source", default="gsm8k_train", choices=["gsm8k_train", "omi_problems"])
    ap.add_argument("--n-questions", type=int, default=7173)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-keep", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open("data/heldout_questions.json") as f:
        held = {qkey(q) for q in json.load(f)}

    qs = load_questions(args.source, held)
    rng = random.Random(args.seed)
    rng.shuffle(qs)
    qs = qs[: args.n_questions]
    print(f"[rft] {len(qs)} questions from {args.source}, k={args.k}", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, enable_prefix_caching=True, seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)

    prompts = [ps.render_prompt(q["question"], None) for q in qs]
    outs = llm.generate(prompts, sp)

    kept = solved = 0
    sysmsg = ps.fewshot_system_message()
    records = []
    for q, o in zip(qs, outs):
        good, seen = [], set()
        for c in o.outputs:
            body = c.text.strip()
            if last_number(body) != q["gold"]:
                continue
            body = re.sub(r"\n*ANSWER:.*$", "", body, flags=re.S).strip()
            if not body or len(body) < 20:
                continue
            h = hashlib.md5(re.sub(r"\s+", " ", body).encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            good.append(body)
        if good:
            solved += 1
        for body in good[: args.max_keep]:
            completion = ps.render_target(body, q["gold"])
            if completion.count(ps.ANSWER_MARKER) != 1 or "boxed" in completion or "####" in completion:
                continue
            records.append({"question": q["question"], "answer": q["gold"], "body": body})
            kept += 1

    rng.shuffle(records)
    n_fs = int(len(records) * args.fewshot_frac)
    with open(args.out, "w") as f:
        for i, r in enumerate(records):
            fewshot = i < n_fs
            f.write(json.dumps({
                "prompt": ps.render_prompt(r["question"], sysmsg if fewshot else None),
                "completion": ps.render_target(r["body"], r["answer"]),
                "source": f"rft:{args.source}", "fewshot": fewshot,
                "question": r["question"], "answer": r["answer"]}) + "\n")

    stats = {"model": args.model, "source": args.source, "questions": len(qs), "k": args.k,
             "temperature": args.temperature, "solved_at_least_once": solved,
             "pass_at_k": solved / max(1, len(qs)), "rows_written": kept, "out": args.out}
    print(json.dumps(stats, indent=2), flush=True)
    if args.stats_out:
        with open(args.stats_out, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
