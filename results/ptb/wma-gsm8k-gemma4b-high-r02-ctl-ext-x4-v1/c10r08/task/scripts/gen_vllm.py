#!/usr/bin/env python3
"""Offline vLLM generation with the grader's exact prompt and stop tokens.

Two jobs:
  * probe    -- score a checkpoint on a jsonl of {id, question, gold}
  * sample   -- draw k solutions per question for rejection-sampling fine-tuning

The prompt is built with templates/gemma3.jinja (the file evaluate.py hands to
vLLM), so what we measure here is what the harness measures, minus inspect's
server plumbing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402


def load_rows(path, limit=None):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="jsonl with question (+gold)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--fewshot", type=int, default=0,
                    help="k-shot prefix drawn from gsm8k TRAIN with inspect's seed-42 recipe")
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    tpl = fmt.template_text()

    system = None
    if args.fewshot:
        system = build_inspect_fewshot(args.fewshot)

    rows = load_rows(args.input, args.limit)
    prompts = [tok.apply_chat_template(fmt.build_messages(r["question"], system),
                                       chat_template=tpl, tokenize=False,
                                       add_generation_prompt=True) for r in rows]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              max_model_len=4096, dtype="bfloat16", seed=args.seed,
              enforce_eager=False, generation_config="vllm")
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_tokens,
                        stop_token_ids=[1, 106], seed=None)
    outs = llm.generate(prompts, sp)

    n_correct = 0
    n_any = 0
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            comps = [c.text for c in o.outputs]
            finish = [c.finish_reason for c in o.outputs]
            gold = str(r.get("gold", ""))
            ok = [bool(gold) and fmt.graded_correct(c, gold) for c in comps]
            n_correct += ok[0] if ok else 0
            n_any += any(ok)
            f.write(json.dumps({"id": r.get("id"), "question": r["question"],
                                "gold": gold, "completions": comps,
                                "correct": ok, "finish": finish}) + "\n")
    n = len(rows)
    summary = {"model": args.model, "n": n, "k": args.k,
               "temperature": args.temperature, "fewshot": args.fewshot,
               "acc_first_sample": n_correct / n if n else 0.0,
               "pass_at_k": n_any / n if n else 0.0}
    print(json.dumps(summary, indent=2), flush=True)
    with open(args.out + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def build_inspect_fewshot(k: int) -> str:
    """Reproduce inspect_evals.gsm8k's system message: hf_dataset(gsm8k train,
    shuffle=True, seed=42, limit=k) -> sample_to_fewshot joined by blank lines."""
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot
    shots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                       sample_fields=record_to_sample, shuffle=True, seed=42, limit=k)
    return "\n\n".join(sample_to_fewshot(s) for s in shots)


if __name__ == "__main__":
    main()
