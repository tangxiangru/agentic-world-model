#!/usr/bin/env python3
"""Rejection-sampling fine-tuning data generation.

Samples k solutions per training problem from the current policy with vLLM,
keeps the ones whose final `ANSWER:` line matches the reference answer, and
writes a deduplicated on-policy SFT file.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import load_dataset

from prepare_data import MATH_PROMPT_TEMPLATE, OMI2_GLOB, clean_answer, fewshot_block

TEMPLATE = "templates/gemma3.jinja"
ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")


def final_answer(text: str):
    ms = ANS_RE.findall(text)
    if not ms:
        return None
    a = ms[-1].replace(",", "")
    try:
        f = float(a)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else str(f)


def load_problems(n_aug: int, seed: int):
    probs = []
    ds = load_dataset("openai/gsm8k", "main", split="train")
    human = []
    for r in ds:
        reasoning, target = r["answer"].split("####")
        t = target.strip().replace(",", "")
        probs.append({"question": r["question"].strip(), "answer": t, "src": "gsm8k"})
        human.append({"question": r["question"].strip(), "solution": reasoning.strip(), "answer": t})

    if n_aug > 0:
        seen = set()
        aug = []
        for f in sorted(glob.glob(OMI2_GLOB)):
            t = pq.read_table(f, columns=["problem", "expected_answer", "problem_source"])
            t = t.filter(pc.is_in(t.column("problem_source"), value_set=pa.array(["augmented_gsm8k"])))
            for r in t.to_pylist():
                q = r["problem"].strip()
                if q in seen or len(q) > 1200:
                    continue
                a = clean_answer(r["expected_answer"])
                if a is None:
                    continue
                seen.add(q)
                aug.append({"question": q, "answer": a, "src": "augmented_gsm8k"})
        random.Random(seed).shuffle(aug)
        probs += aug[:n_aug]
    return probs, human


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckpt/sft_v1")
    ap.add_argument("--out", default="data/rft.jsonl")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n-aug", type=int, default=15000)
    ap.add_argument("--keep-per-problem", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.88)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    probs, human = load_problems(args.n_aug, args.seed)
    print(f"[rft] {len(probs)} problems x k={args.k}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    ct = open(TEMPLATE).read()

    prompts, meta = [], []
    n_fs = int(len(probs) * args.fewshot_frac)
    for i, p in enumerate(probs):
        msgs = []
        if i < n_fs:
            msgs.append({"role": "system", "content": fewshot_block(rng.sample(human, rng.choice([2, 4, 8, 10])))})
        msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=p["question"])})
        txt = tok.apply_chat_template(msgs, chat_template=ct, tokenize=False, add_generation_prompt=True)
        # the rendered template already contains <bos>; do not let vLLM add a second one
        prompts.append({"prompt_token_ids": tok(txt, add_special_tokens=False)["input_ids"]})
        meta.append((i, msgs[0]["content"] if i < n_fs else None))

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=4096,
        enforce_eager=False,
        dtype="bfloat16",
        enable_prefix_caching=True,
        max_num_seqs=512,
        disable_log_stats=True,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temp,
        top_p=0.95,
        top_k=64,
        max_tokens=args.max_tokens,
        seed=args.seed,
        # <end_of_turn> is what the chat template trains the model to emit; the
        # offline API does not pick it up from the generation config.
        stop_token_ids=[1, 106],
    )
    outs = llm.generate(prompts, sp)

    kept = defaultdict(list)
    n_corr = n_tot = 0
    solved = 0
    for (i, sysmsg), o in zip(meta, outs):
        gold = probs[i]["answer"]
        good = []
        for c in o.outputs:
            n_tot += 1
            txt = c.text.strip()
            if final_answer(txt) == gold and txt.count("ANSWER:") == 1:
                n_corr += 1
                good.append(txt)
        if good:
            solved += 1
        # dedup by normalised text, prefer shorter (less rambling) solutions
        seen, uniq = set(), []
        for t in sorted(good, key=len):
            key = re.sub(r"\s+", " ", t)[:400]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(t)
        kept[i] = (uniq[: args.keep_per_problem], sysmsg)

    n_out = 0
    with open(args.out, "w") as f:
        for i, (sols, sysmsg) in kept.items():
            for s in sols:
                f.write(
                    json.dumps(
                        {
                            "system": sysmsg,
                            "user": MATH_PROMPT_TEMPLATE.format(prompt=probs[i]["question"]),
                            "completion": s,
                            "question": probs[i]["question"],
                            "answer": probs[i]["answer"],
                            "source": "rft_" + probs[i]["src"],
                        }
                    )
                    + "\n"
                )
                n_out += 1
    print(
        f"[rft] pass@1 {n_corr/max(1,n_tot):.3f}  solved-any {solved/len(probs):.3f}  wrote {n_out} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
