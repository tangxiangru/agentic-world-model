"""Rejection-sampling fine-tuning data: sample k solutions per training question
from the current checkpoint, keep the ones whose final answer is right.

Questions come from the GSM8K *train* split and from OpenMathInstruct-2's
augmented_gsm8k problems (also train-seeded). No test item is read.

The answer is extracted with the grader's own rule (last number in the
completion, see inspect_ai.scorer._common.match_str with location='end'), so a
kept sample is one the grader would have marked correct.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402
from build_data import OMI_DIR, NUM_RE, norm_q, fewshot_pool  # noqa: E402

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
WORD = re.compile(r"\s+")


def last_number(text: str):
    """The grader's rule: last whitespace-delimited token that parses as a number."""
    t = text.replace(",", "").replace("$", "")
    for w in reversed(WORD.split(t.strip())):
        w = w.strip(".):;*")
        if w.replace(".", "").replace("-", "").isnumeric():
            try:
                return float(w)
            except ValueError:
                return None
    return None


def same(a: str, b: str) -> bool:
    x, y = last_number(a), last_number(b)
    return x is not None and y is not None and abs(x - y) < 1e-6


def gsm8k_train_questions():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    return [{"q": r["question"].strip(), "a": r["answer"].rsplit("####", 1)[1].strip().replace(",", "")}
            for r in ds]


def omi_questions(n, shards=range(0, 10)):
    seen = set()
    out = []
    for sh in shards:
        p = os.path.join(OMI_DIR, f"train-{sh:05d}-of-00032.parquet")
        if not os.path.exists(p):
            continue
        for batch in pq.ParquetFile(p).iter_batches(
                batch_size=20000, columns=["problem", "expected_answer", "problem_source"]):
            for r in batch.to_pylist():
                if r["problem_source"] != "augmented_gsm8k":
                    continue
                a = (r["expected_answer"] or "").strip()
                if not NUM_RE.match(a):
                    continue
                k = norm_q(r["problem"])
                if k in seen:
                    continue
                seen.add(k)
                out.append({"q": r["problem"].strip(), "a": a})
                if len(out) >= n:
                    return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-omi", type=int, default=25000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--fewshot-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    qs = gsm8k_train_questions()
    if args.n_omi:
        qs += omi_questions(args.n_omi)
    print(f"[rft] {len(qs)} questions x k={args.k}", flush=True)

    tok = AutoTokenizer.from_pretrained(SNAP)
    prompts = [TokensPrompt(prompt_token_ids=tok.encode(fmt.render_prompt(x["q"]),
                                                        add_special_tokens=False)) for x in qs]
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, enforce_eager=False, seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95, top_k=64,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=None)
    outs = llm.generate(prompts, sp)

    import random
    rng = random.Random(args.seed)
    pool = fewshot_pool()
    kept = 0
    solved = 0
    per_q = Counter()
    stats = defaultdict(int)
    with open(args.out, "w") as f:
        for x, o in zip(qs, outs):
            texts = []
            for c in o.outputs:
                if c.finish_reason != "stop":
                    stats["truncated"] += 1
                    continue
                t = c.text.strip()
                if not same(t, x["a"]):
                    stats["wrong"] += 1
                    continue
                texts.append(t)
            if texts:
                solved += 1
            seen = set()
            uniq = []
            for t in texts:
                key = re.sub(r"\s+", " ", t)[:400]
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(t)
            uniq.sort(key=len)  # prefer the shortest correct chains
            for t in uniq[: args.keep_per_question]:
                if fmt.ANSWER_MARKER not in t:
                    stats["no_marker"] += 1
                    continue
                if t.count(fmt.ANSWER_MARKER) != 1:
                    stats["multi_marker"] += 1
                    continue
                system = None
                if rng.random() < args.fewshot_frac:
                    system = "\n\n".join(rng.sample(pool, rng.randint(1, 8)))
                f.write(json.dumps({
                    "prompt": fmt.render_prompt(x["q"], system),
                    "completion": fmt.render_completion(t),
                    "text": x["q"] + "\n\n" + t,
                    "src": "rft_self",
                }) + "\n")
                kept += 1
                per_q[x["q"][:60]] += 1
    print(f"[rft] questions solved at least once: {solved}/{len(qs)} ({solved/len(qs):.3f})")
    print(f"[rft] rows kept: {kept}  drops: {dict(stats)}")


if __name__ == "__main__":
    main()
