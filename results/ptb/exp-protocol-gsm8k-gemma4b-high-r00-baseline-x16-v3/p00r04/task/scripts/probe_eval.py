"""Local dev scorer: same prompt, same chat template, same matcher as the grader,
but run on GSM8K *train* hold-outs (data/probe250.jsonl) so no benchmark test item
is ever inspected.  Used for iteration; the official evaluate.py is the arbiter.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import fmt as F  # noqa: E402


def score_end_numeric(completion: str, gold: str) -> bool:
    """Mirror inspect_ai.scorer.match(location='end', numeric=True)."""
    from inspect_ai.scorer._common import match_str

    _, ok = match_str(completion, gold, location="end", ignore_case=True, numeric=True)
    return bool(ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data", default="data/probe250.jsonl")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--fewshot", type=int, default=1, help="1 = use the grader's 10-shot system message")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.data)][: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model_path)
    prompts = [F.render_prompt(tok, r["question"], fewshot=bool(args.fewshot)) for r in rows]

    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        dtype="bfloat16",
        enforce_eager=False,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=1.0 if args.temperature == 0 else 0.95,
        max_tokens=args.max_tokens,
        n=args.n,
        stop_token_ids=[1, 106],
    )
    outs = llm.generate(prompts, sp)

    n_ok = 0
    recs = []
    for r, o in zip(rows, outs):
        texts = [c.text for c in o.outputs]
        oks = [score_end_numeric(t, r["gold"]) for t in texts]
        ok = oks[0]
        n_ok += int(ok)
        recs.append(
            {
                "id": r["id"],
                "gold": r["gold"],
                "correct": ok,
                "maj_correct": sum(oks) / len(oks),
                "completion": texts[0],
            }
        )
    acc = n_ok / len(rows)
    print(f"PROBE accuracy {acc:.4f}  ({n_ok}/{len(rows)})  model={args.model_path} fewshot={args.fewshot}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"accuracy": acc, "n": len(rows), "model": args.model_path,
                       "fewshot": bool(args.fewshot), "samples": recs}, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
