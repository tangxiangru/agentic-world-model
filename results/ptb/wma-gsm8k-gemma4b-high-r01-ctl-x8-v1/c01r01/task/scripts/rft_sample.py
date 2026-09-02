#!/usr/bin/env python3
"""Rejection-sampling data generation from a fine-tuned checkpoint.

Samples k completions per problem with vLLM, keeps the ones whose final
'ANSWER: n' equals the problem's reference answer, and writes them in the same
prompt/completion form build_data.py emits (stop token included).

Problems come from GSM8K TRAIN and from OpenMathInstruct-2's gsm8k-sourced
problems (both of which carry a reference answer).  The GSM8K test split is
never read.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, "/home/ben/task/scripts")
import fmt  # noqa: E402
from build_data import GSM_TRAIN, OMI_GLOB, norm_answer  # noqa: E402

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,\.]+)")


def extract(c: str):
    m = ANS_RE.findall(c)
    if not m:
        return None
    return norm_answer(m[-1])


def load_problems(n_omi: int, seed: int):
    import pyarrow.parquet as pq
    probs = []
    for r in pq.read_table(GSM_TRAIN).to_pylist():
        parts = r["answer"].split("####")
        a = norm_answer(parts.pop().strip())
        if a is not None:
            probs.append((r["question"], a, "gsm8k_train"))
    seen = set(q for q, _, _ in probs)
    omi = []
    for f in sorted(glob.glob(OMI_GLOB)):
        t = pq.read_table(f, columns=["problem", "expected_answer", "problem_source"])
        for r in t.to_pylist():
            if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                continue
            a = norm_answer(r["expected_answer"] or "")
            if a is None or r["problem"] in seen or len(r["problem"]) > 1200:
                continue
            seen.add(r["problem"])
            omi.append((r["problem"], a, "omi_gsm8k"))
    random.Random(seed).shuffle(omi)
    return probs + omi[:n_omi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-omi", type=int, default=30000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-fewshot-small", type=float, default=0.08)
    ap.add_argument("--p-fewshot-official", type=float, default=0.04)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(fmt.BASE_SNAPSHOT)
    problems = load_problems(a.n_omi, a.seed)
    print(f"{len(problems)} problems", flush=True)
    prompts = [fmt.render(tok, None, q, None)[0] for q, _, _ in problems]

    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_memory_utilization,
              max_model_len=1024, dtype="bfloat16", seed=a.seed,
              enforce_eager=False, generation_config="vllm")
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p,
                        top_k=a.top_k, max_tokens=a.max_tokens, seed=None,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    # same prompt-variant mixture as build_data.py: sampling is done zero-shot
    # (cheap) but the training rows keep the grader's few-shot prefixes on a
    # slice, so prompt robustness is not lost in the RFT round.
    official_sys = open("/home/ben/task/data/official_fewshot_system.txt").read()
    shot_pool = []
    import pyarrow.parquet as pq
    for r in pq.read_table(GSM_TRAIN).to_pylist():
        parts = r["answer"].split("####")
        tgt = parts.pop().strip()
        shot_pool.append(fmt.fewshot_block(r["question"], "####".join(parts).strip(), tgt))
    wrng = random.Random(a.seed + 1)

    kept = 0
    stats = defaultdict(int)
    solved_hist = defaultdict(int)
    with open(a.out, "w") as f, open(a.out.replace(".jsonl", "_decon.jsonl"), "w") as fd:
        for (q, gold, src), o, p in zip(problems, outs, prompts):
            good = []
            for cand in o.outputs:
                txt = cand.text
                if cand.finish_reason != "stop":
                    stats["unterminated"] += 1
                    continue
                got = extract(txt)
                if got is None:
                    stats["no_answer"] += 1
                    continue
                if got != gold:
                    stats["wrong"] += 1
                    continue
                t = txt.strip()
                if t.count("ANSWER:") != 1:
                    stats["multi_marker"] += 1
                    continue
                good.append(t)
            solved_hist[len(good)] += 1
            # prefer shorter correct solutions, dedup exact
            good = sorted(set(good), key=len)[: a.max_per_problem]
            for t in good:
                u = wrng.random()
                if u < a.p_fewshot_official:
                    pr = fmt.render(tok, official_sys, q, None)[0]
                elif u < a.p_fewshot_official + a.p_fewshot_small:
                    pr = fmt.render(tok, "\n\n".join(wrng.sample(shot_pool, wrng.randint(1, 4))), q, None)[0]
                else:
                    pr = p
                f.write(json.dumps({"prompt": pr, "completion": t + fmt.STOP_TOKEN,
                                    "src": "rft_" + src}) + "\n")
                fd.write(json.dumps({"question": q, "answer": t}) + "\n")
                kept += 1
    print(f"kept {kept} rows -> {a.out}", flush=True)
    print("reject stats", dict(stats), flush=True)
    print("n correct per problem", dict(sorted(solved_hist.items())), flush=True)
    n_any = sum(v for k, v in solved_hist.items() if k > 0)
    print(f"pass@{a.k} = {n_any/len(problems):.3f}", flush=True)


if __name__ == "__main__":
    main()
