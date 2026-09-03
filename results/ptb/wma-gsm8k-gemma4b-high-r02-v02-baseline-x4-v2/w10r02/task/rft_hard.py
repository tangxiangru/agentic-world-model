#!/usr/bin/env python3
"""Second-pass rejection sampling with a bigger budget on the questions the
model rarely or never solves.

Reads the raw generations of the first pass, keeps the questions with <= --max-hits
correct samples out of 4, and re-samples them k times at a higher temperature.
Same train-derived question pool as pass 1 - no test item is ever involved.
"""
from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer

from build_data import EOT, render_prompt
from rft_sample import SNAPSHOT, last_number


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hits", type=int, default=1)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.97)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    hard = []
    for line in open(args.raw):
        r = json.loads(line)
        hits = sum(1 for g in r["gens"] if last_number(g.strip()) == r["a"])
        if hits <= args.max_hits:
            hard.append(r)
    print(f"{len(hard)} hard questions (<= {args.max_hits} correct of 4)", flush=True)

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              max_model_len=2560, dtype="bfloat16", seed=args.seed)
    sp = SamplingParams(n=args.samples, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    prompts = [
        {"prompt_token_ids": tok(render_prompt(h["q"], None), add_special_tokens=False)["input_ids"]}
        for h in hard
    ]
    outs = llm.generate(prompts, sp)

    kept, solved, n_tot, n_ok = [], 0, 0, 0
    for h, out in zip(hard, outs):
        texts = []
        for c in out.outputs:
            n_tot += 1
            t = c.text.strip()
            if last_number(t) != h["a"] or t.count("ANSWER:") != 1:
                continue
            n_ok += 1
            texts.append(t)
        solved += int(bool(texts))
        for t in sorted(set(texts), key=len)[: args.keep_per_question]:
            kept.append({"prompt": render_prompt(h["q"], None), "completion": t + EOT,
                         "src": "rfthard:" + h["src"], "answer": h["a"]})
    print(f"pass@1 {n_ok/max(n_tot,1):.3f}  newly solved {solved}/{len(hard)}  kept {len(kept)}")
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
