#!/usr/bin/env python3
"""Sample k completions per training problem and dump all of them (with the
extracted answer of each) for downstream RFT / self-consistency data building."""
from __future__ import annotations

import argparse
import json
import random

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from gen_samples import extract, load_problems, norm_num
from prep_data import MATH_PROMPT_TEMPLATE
from train_sft import BASE, build_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/samples.jsonl")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n-aug", type=int, default=15000)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    probs = load_problems(args, rng)
    print(f"{len(probs)} problems, k={args.k}")

    tok = AutoTokenizer.from_pretrained(BASE)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=1024,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temp,
        top_p=0.95,
        top_k=64,
        max_tokens=args.max_tokens,
        stop_token_ids=[106, 1],
        seed=args.seed,
    )
    prompts = [
        TokensPrompt(
            prompt_token_ids=tok(
                build_prompt("", MATH_PROMPT_TEMPLATE.format(prompt=p)),
                add_special_tokens=False,
            )["input_ids"]
        )
        for p, _, _ in probs
    ]
    outs = llm.generate(prompts, sp)

    n_any = 0
    with open(args.out, "w") as f:
        for (problem, gold, src), o in zip(probs, outs):
            g = norm_num(gold)
            cands = []
            for c in o.outputs:
                if c.finish_reason != "stop":
                    continue
                txt = c.text.strip()
                a = extract(txt)
                cands.append({"text": txt, "ans": a, "ok": a is not None and g is not None and a == g})
            if any(c["ok"] for c in cands):
                n_any += 1
            f.write(
                json.dumps(
                    {"problem": problem, "answer": gold, "source": src, "cands": cands}
                )
                + "\n"
            )
    print(f"problems with >=1 correct: {n_any}/{len(probs)}")


if __name__ == "__main__":
    main()
