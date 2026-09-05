#!/usr/bin/env python3
"""How much does the grader's 10-shot system prefix cost a zero-shot-trained model?

Reproduces inspect_evals/gsm8k's system message exactly (same hf_dataset call,
shuffle seed 42, limit 10, TRAIN split) and scores the same held-out questions
with and without it. Held-out questions come from the OpenMathInstruct-2
gsm8k-family pool minus everything in the SFT mix. No test item is read.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re

import pyarrow.parquet as pq

OMI2 = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_[12]M-*.parquet"
PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def eval_system_message() -> str:
    """Byte-identical to what inspect_evals.gsm8k builds for fewshot=10."""
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

    fewshots = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=42,
        limit=10,
    )
    return "\n\n".join(sample_to_fewshot(s) for s in fewshots)


def num(s):
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def last_number(t: str):
    words = re.split(r"\s+", t.strip())
    for w in reversed(words):
        w2 = w.strip("*$.,:;()")
        if w2.replace(".", "").replace(",", "").lstrip("-").isnumeric():
            return num(w2)
    return None


def held_out(n: int, exclude: set[str], seed: int, include_math: bool = False):
    """Questions no training mix has seen. The gsm8k-family pool is nearly
    exhausted by the SFT mixes, so it can be topped up with the math-family
    problems, which no card trains on. Returns (question, gold, family)."""
    gsm, mth = [], []
    for f in sorted(glob.glob(OMI2)):
        d = pq.read_table(f, columns=["problem", "expected_answer", "problem_source"]).to_pydict()
        for q, a, s in zip(d["problem"], d["expected_answer"], d["problem_source"]):
            g = num(a)
            q = q.strip()
            if g is None or q in exclude:
                continue
            (gsm if s in ("gsm8k", "augmented_gsm8k") else mth).append((q, g, "gsm8k"
                if s in ("gsm8k", "augmented_gsm8k") else "math"))
    def dedup(rows):
        seen, uniq = set(), []
        for r in rows:
            if r[0] in seen:
                continue
            seen.add(r[0])
            uniq.append(r)
        random.Random(seed).shuffle(uniq)
        return uniq
    gsm, mth = dedup(gsm), dedup(mth)
    out = gsm[:n]
    if include_math and len(out) < n:
        out = out + mth[: n - len(out)]
    print(f"held-out pool: {sum(1 for r in out if r[2] == 'gsm8k')} gsm8k-family, "
          f"{sum(1 for r in out if r[2] == 'math')} math-family")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--exclude-file", default="/home/ben/task/data/sft_v1.jsonl",
                    help="comma-separated jsonl files whose 'question' values are held out")
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--include-math", action="store_true")
    a = ap.parse_args()

    excl = set()
    for path in a.exclude_file.split(","):
        for line in open(path):
            q = json.loads(line).get("question")
            if q:
                excl.add(q.strip())
    qs = held_out(a.n, excl, seed=1234, include_math=a.include_math)
    print(f"{len(qs)} held-out questions (excluded {len(excl)})")

    sysmsg = eval_system_message()
    print(f"system message: {len(sysmsg)} chars")

    from vllm import LLM, SamplingParams

    tmpl = open("/home/ben/task/templates/gemma3.jinja").read()
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_frac, max_model_len=4096,
              enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)

    res = {}
    for name, use_sys in (("zeroshot", False), ("tenshot", True)):
        convs = []
        for q, _, _fam in qs:
            m = [{"role": "user", "content": PROMPT_TEMPLATE.format(prompt=q)}]
            if use_sys:
                m = [{"role": "system", "content": sysmsg}] + m
            convs.append(m)
        outs = llm.chat(convs, sp, chat_template=tmpl)
        ok = stopped = 0
        gok = gn = 0
        lens = []
        for (q, g, fam), o in zip(qs, outs):
            t = o.outputs[0].text
            lens.append(len(t))
            if o.outputs[0].finish_reason == "stop":
                stopped += 1
            v = last_number(t)
            hit = v is not None and abs(v - g) < 1e-6
            ok += hit
            if fam == "gsm8k":
                gn += 1
                gok += hit
        res[name] = {
            "accuracy": round(ok / len(qs), 4),
            "gsm8k_family_accuracy": round(gok / gn, 4) if gn else None,
            "gsm8k_family_n": gn,
            "stop_share": round(stopped / len(qs), 4),
            "median_chars": sorted(lens)[len(lens) // 2],
        }
        print(name, res[name])
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump({"model": a.model, "n": len(qs), "results": res}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
