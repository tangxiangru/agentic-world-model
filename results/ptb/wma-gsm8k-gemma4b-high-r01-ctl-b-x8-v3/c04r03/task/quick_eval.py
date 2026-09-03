#!/usr/bin/env python3
"""Fast offline dev eval: same prompt rendering and same scorer as the harness,
run through vLLM's batch API instead of the inspect server so 600 items cost
about as much wall clock as 150 through evaluate.py.

Used only as a diagnostic / checkpoint-ranking signal. The card's protocol
number always comes from evaluate.py --limit 150.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from inspect_ai.scorer._common import match_str
from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE

TASK = Path(__file__).resolve().parent
TEMPLATE = (TASK / "templates" / "gemma3.jinja").read_text()


def harness_fewshot(k: int, seed: int = 42) -> str:
    """Rebuild the harness's system message exactly: hf_dataset(openai/gsm8k,
    split=train, shuffle=True, seed=42, limit=k) -> sample_to_fewshot joined."""
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot
    ds = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                    sample_fields=record_to_sample, shuffle=True, seed=seed, limit=k)
    return "\n\n".join(sample_to_fewshot(s) for s in ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/privdev.jsonl")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--fewshot", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)][: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)
    prefix = (harness_fewshot(args.fewshot) + "\n\n") if args.fewshot else ""

    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": prefix + MATH_PROMPT_TEMPLATE.format(prompt=r["question"])}],
        chat_template=TEMPLATE, tokenize=False, add_generation_prompt=True) for r in rows]

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              max_model_len=4096, seed=0)
    sp = SamplingParams(temperature=args.temp, top_p=1.0, top_k=-1,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    n_ok = n_fmt = n_cut = 0
    dump = []
    for r, o in zip(rows, outs):
        txt = o.outputs[0].text
        _, ok = match_str(value=txt, target=r["answer"], location="end",
                          ignore_case=True, numeric=True)
        n_ok += ok
        if "ANSWER:" not in txt:
            n_fmt += 1
        if o.outputs[0].finish_reason == "length":
            n_cut += 1
        dump.append({"q": r["question"], "gold": r["answer"], "ok": bool(ok),
                     "finish": o.outputs[0].finish_reason, "out": txt})
    n = len(rows)
    res = {"model": args.model, "n": n, "fewshot": args.fewshot,
           "accuracy": n_ok / n, "no_answer_marker": n_fmt / n, "length_capped": n_cut / n}
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        with open(args.out + ".samples.jsonl", "w") as fh:
            for d in dump:
                fh.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()
