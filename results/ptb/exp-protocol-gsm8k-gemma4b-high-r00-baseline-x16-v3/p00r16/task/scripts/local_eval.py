"""Fast local probe eval: same prompt + same scorer as evaluate.py, but on a
held-out slice of the GSM8K *train* split (never the benchmark test set).

Used as the `diagnostic` protocol in the cards; the official comparator is
always evaluate.py --limit N.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompt_utils as P  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data", default="/home/ben/task/data/probe300.jsonl")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--no-fewshot", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=1, help="samples per question (maj@n)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.data)][: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model_path)
    prompts = [P.eval_prompt(tok, r["question"], fewshot=not args.no_fewshot) for r in rows]

    llm = LLM(model=args.model_path, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=4096, dtype="bfloat16", enforce_eager=False)
    sp = SamplingParams(temperature=args.temperature, top_p=1.0 if args.temperature == 0 else 0.95,
                        max_tokens=args.max_tokens, n=args.n, seed=0)
    outs = llm.generate(prompts, sp)

    n_ok = 0
    recs = []
    for r, o in zip(rows, outs):
        texts = [c.text for c in o.outputs]
        oks = [P.grade(t, r["gold"]) for t in texts]
        ok = oks[0]
        n_ok += ok
        recs.append({"id": r["id"], "gold": r["gold"], "correct": bool(ok),
                     "finished": o.outputs[0].finish_reason,
                     "n_tokens": len(o.outputs[0].token_ids),
                     "completion": texts[0]})
    acc = n_ok / len(rows)
    summary = {"model": args.model_path, "data": args.data, "n": len(rows),
               "accuracy": acc, "fewshot": not args.no_fewshot,
               "temperature": args.temperature}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "samples": recs}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
