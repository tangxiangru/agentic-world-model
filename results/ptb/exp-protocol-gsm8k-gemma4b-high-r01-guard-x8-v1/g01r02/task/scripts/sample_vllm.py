#!/usr/bin/env python3
"""Sample completions from a checkpoint using the grader's exact prompt, and
grade them with the grader's exact rule (inspect_ai match(numeric=True)).

Used for two things: a cheap dev probe on held-out GSM8K *train* items, and
rejection sampling for RFT. Prompts are built by scripts/build_data.render_prompt,
which was verified byte-identical to templates/gemma3.jinja.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import render_prompt  # noqa: E402

from inspect_ai.scorer._common import match_str  # noqa: E402


def load_eval_fewshots():
    """The exact 10 shots inspect_evals/gsm8k puts in the system slot."""
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample
    ds = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                    sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
    return [(s.input, s.metadata["reasoning"], s.target) for s in ds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="jsonl with id/question/gold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--shots", type=int, default=10, choices=[0, 10])
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.input)]
    if args.limit:
        items = items[:args.limit]
    shots = load_eval_fewshots() if args.shots else []

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [TokensPrompt(prompt_token_ids=tok(render_prompt(it["question"], shots),
                                                 add_special_tokens=False)["input_ids"])
               for it in items]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_len, enforce_eager=False, dtype="bfloat16")
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=0,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    n_ok = n_tot = 0
    with open(args.out, "w") as f:
        for it, o in zip(items, outs):
            samples = []
            for c in o.outputs:
                text = c.text
                _, ok = match_str(text, it["gold"], location="end",
                                  ignore_case=True, numeric=True)
                samples.append({"text": text, "correct": bool(ok)})
                n_ok += bool(ok)
                n_tot += 1
            f.write(json.dumps({"id": it["id"], "question": it["question"],
                                "gold": it["gold"], "samples": samples}) + "\n")
    print(f"[sample] {n_ok}/{n_tot} correct = {n_ok / max(1, n_tot):.4f}", flush=True)
    solved = sum(1 for l in open(args.out) if any(s["correct"] for s in json.loads(l)["samples"]))
    print(f"[sample] questions with >=1 correct: {solved}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
