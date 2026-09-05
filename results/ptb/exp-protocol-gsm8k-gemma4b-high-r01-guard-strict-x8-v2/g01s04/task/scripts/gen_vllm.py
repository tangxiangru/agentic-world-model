#!/usr/bin/env python3
"""Sample completions with vLLM under the grader's exact prompt + template.

Two uses:
  --mode eval   : greedy, 1 sample/question, prints accuracy (internal dev)
  --mode sample : temperature sampling, k samples/question, writes every
                  completion with a correctness flag (rejection-sampling input)

The prompt is built with the same MATH_PROMPT_TEMPLATE and the same
templates/gemma3.jinja that evaluate.py passes to vLLM. --fewshot 10 reproduces
the harness's 10-shot system prefix (drawn from gsm8k *train*, seed 42, the
same call inspect_evals makes).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    END_OF_TURN,
    SNAPSHOT,
    build_messages,
    is_correct,
    load_tokenizer,
    sample_to_fewshot,
)

CALC_FREE = __import__("re").compile(r"<<[^>]*>>")


def harness_fewshots(n: int = 10, seed: int = 42) -> list[str]:
    """inspect_evals builds these with hf_dataset(..., shuffle=True, seed=42,
    limit=10) over gsm8k train. Reproduce with the same datasets shuffle."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    out = []
    for i in range(n):
        r = ds[i]
        body, _, tail = r["answer"].rpartition("####")
        # keep the <<a*b=c>> annotations: inspect_evals passes the raw gsm8k
        # reasoning through sample_to_fewshot untouched
        out.append(sample_to_fewshot(r["question"].strip(), body.strip(), tail.strip()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SNAPSHOT)
    ap.add_argument("--questions", required=True, help="jsonl with question/answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["eval", "sample"], default="eval")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--fewshot", type=int, default=10)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    rows = [json.loads(x) for x in open(args.questions)]
    if args.limit > 0:
        rows = rows[: args.limit]

    tok = load_tokenizer()
    shots = harness_fewshots(args.fewshot) if args.fewshot else None

    prompts = []
    for r in rows:
        msgs = build_messages(r["question"], shots)
        prompts.append(
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )
    print("example prompt tokens:", len(tok(prompts[0])["input_ids"]), flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop=[END_OF_TURN, "<start_of_turn>"],
        seed=0 if args.mode == "eval" else None,
    )
    outs = llm.generate(prompts, sp)

    n_ok = 0
    n_fmt = 0
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            comps = []
            any_ok = False
            for c in o.outputs:
                text = c.text.strip()
                ok = is_correct(text, str(r["answer"]))
                any_ok |= ok
                fmt = text.rstrip().split("\n")[-1].strip().startswith("ANSWER:")
                comps.append({"text": text, "correct": ok, "fmt": fmt,
                              "stop": c.finish_reason})
            n_ok += int(comps[0]["correct"]) if args.mode == "eval" else int(any_ok)
            n_fmt += sum(c["fmt"] for c in comps) / len(comps)
            f.write(json.dumps({"id": r.get("id"), "question": r["question"],
                                "answer": r["answer"], "completions": comps}) + "\n")

    label = "accuracy" if args.mode == "eval" else "pass@k"
    print(f"{label}: {n_ok / len(rows):.4f}  ({n_ok}/{len(rows)})")
    print(f"format_rate: {n_fmt / len(rows):.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
