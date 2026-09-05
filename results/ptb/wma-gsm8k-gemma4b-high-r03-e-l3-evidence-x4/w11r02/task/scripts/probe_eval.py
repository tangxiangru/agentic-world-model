"""Offline probe on held-out gsm8k TRAIN items, graded exactly like the harness.

Not a substitute for evaluate.py - the protocol number always comes from
evaluate.py --limit 150. This is the cheap diagnostic: same prompt (fmt.py, the
grader's own template), same scorer (inspect's match(numeric=True, location=
"end")), 300 items the training corpus never saw, one vLLM process, no server.

  python scripts/probe_eval.py --model <dir> --out analysis/<name>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/probe300.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-shot", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--n", type=int, default=1, help="samples per item")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--save-completions", default=None)
    args = ap.parse_args()

    from inspect_ai.scorer._common import match_str
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [fmt.render_prompt(r["question"], args.n_shot, tok) for r in rows]

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        generation_config="vllm",
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.n,
        stop_token_ids=[1, 106],
        seed=0 if args.temperature > 0 else None,
    )
    outs = llm.generate(prompts, sp)

    n_ok = 0
    n_trunc = 0
    lens = []
    recs = []
    for r, o in zip(rows, outs):
        texts = [c.text for c in o.outputs]
        for c in o.outputs:
            lens.append(len(c.token_ids))
            if c.finish_reason == "length":
                n_trunc += 1
        oks = [match_str(t, r["answer"], location="end", numeric=True)[1] for t in texts]
        ok = oks[0]
        n_ok += bool(ok)
        recs.append({
            "id": r["id"], "gold": r["answer"], "correct": bool(ok),
            "pass_any": bool(any(oks)), "n_correct": int(sum(oks)),
            "completions": texts if args.save_completions else texts[:1],
        })

    lens.sort()
    summary = {
        "model": args.model,
        "n": len(rows),
        "n_shot": args.n_shot,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "samples_per_item": args.n,
        "accuracy": n_ok / len(rows),
        "pass_any": sum(r["pass_any"] for r in recs) / len(rows),
        "truncated": n_trunc,
        "out_tokens_p50": lens[len(lens) // 2],
        "out_tokens_p99": lens[int(len(lens) * 0.99)],
        "out_tokens_max": lens[-1],
    }
    print(json.dumps(summary, indent=2), flush=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "items": recs}, f, indent=2)
    if args.save_completions:
        with open(args.save_completions, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
