"""Offline vLLM generation used for (a) probe evaluation on held-out GSM8K
*train* items and (b) rejection-sampling data generation.

Prompts are rendered with harness_format.render_prompt, i.e. byte-identical to
what evaluate.py/vLLM produce for the graded turn.
"""
from __future__ import annotations

import argparse
import json
import os

from transformers import AutoTokenizer

import harness_format as hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="jsonl with question/gold (or answer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--fewshot", type=int, default=10, help="0 = zero-shot prompt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    items = [json.loads(l) for l in open(args.input)]
    if args.limit:
        items = items[: args.limit]
    for i, it in enumerate(items):
        it.setdefault("id", str(i))
        if "gold" not in it:
            it["gold"] = it.get("answer")

    tok = AutoTokenizer.from_pretrained(args.model)
    prefix = hf.fewshot_prefix(args.fewshot) if args.fewshot else None
    prompts = [hf.render_prompt(tok, it["question"], prefix=prefix) for it in items]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=0 if args.temperature == 0 else None,
    )
    outs = llm.generate(prompts, sp)

    n_corr = 0
    n_any = 0
    with open(args.out, "w") as f:
        for it, o in zip(items, outs):
            comps = [c.text for c in o.outputs]
            stops = [c.finish_reason for c in o.outputs]
            corr = [hf.is_correct(c, it["gold"]) for c in comps]
            n_corr += sum(corr) / len(corr)
            n_any += 1 if any(corr) else 0
            f.write(json.dumps({
                "id": it["id"], "question": it["question"], "gold": it["gold"],
                "completions": comps, "correct": corr, "finish": stops,
            }) + "\n")
    n = len(items)
    print(f"MODEL {args.model}")
    print(f"items={n} mean_acc={n_corr / n:.4f} pass@{args.n_samples}={n_any / n:.4f} "
          f"temp={args.temperature} fewshot={args.fewshot}")
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump({"model": args.model, "n": n, "mean_acc": n_corr / n,
                   f"pass@{args.n_samples}": n_any / n, "temperature": args.temperature,
                   "fewshot": args.fewshot}, f, indent=2)


if __name__ == "__main__":
    main()
