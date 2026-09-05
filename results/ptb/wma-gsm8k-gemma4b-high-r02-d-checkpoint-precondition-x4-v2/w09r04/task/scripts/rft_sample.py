"""Rejection-sampling fine-tuning data: sample k solutions per training-pool
question from a trained checkpoint, keep the ones whose ANSWER line matches gold.

Only data/gsm8k_trainpool.jsonl (GSM8K *train* rows, minus the 500 held-out dev
rows) is sampled. No benchmark test item is read.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (ANSWER_MARKER, MATH_PROMPT_TEMPLATE, STOP_TOKEN,
                    load_tokenizer, render_prompt)

ROOT = Path(__file__).resolve().parent.parent
ANSWER_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")


def extract(text: str) -> str | None:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None
    m = ANSWER_RE.match(lines[-1].strip())
    return m.group(1).replace(",", "") if m else None


def same(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a == b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", default=str(ROOT / "data" / "gsm8k_trainpool.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-keep", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    tok = load_tokenizer()
    qs = [json.loads(l) for l in Path(args.questions).open()]
    if args.limit > 0:
        qs = qs[: args.limit]
    prompts = [render_prompt(tok, [{"role": "user", "content":
                                    MATH_PROMPT_TEMPLATE.format(prompt=q["question"])}])
               for q in qs]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=4096, seed=args.seed, enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95, top_k=64,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    n_solved = n_rows = 0
    per_q_correct = []
    with Path(args.out).open("w") as fh:
        for q, o in zip(qs, outs):
            cands = []
            for c in o.outputs:
                text = c.text.strip()
                pred = extract(text)
                if pred is not None and same(pred, q["gold"]) and text not in cands:
                    cands.append(text)
            per_q_correct.append(len(cands))
            if not cands:
                continue
            n_solved += 1
            cands.sort(key=len)                      # prefer the tersest correct chains
            keep = cands[: args.max_keep]
            for target in keep:
                n_rows += 1
                target = target + STOP_TOKEN   # vLLM strips the stop token from .text
                user = MATH_PROMPT_TEMPLATE.format(prompt=q["question"])
                fh.write(json.dumps({
                    "messages": [{"role": "user", "content": user}],
                    "completion": target,
                    "text": q["question"] + "\n" + target,
                    "qid": q["id"],
                }) + "\n")

    solve_rate = sum(1 for c in per_q_correct if c) / len(qs)
    pass1 = sum(per_q_correct) / (len(qs) * args.k)
    stats = {"n_questions": len(qs), "k": args.k, "pass_at_k": solve_rate,
             "mean_pass_rate": pass1, "n_solved": n_solved, "n_rows": n_rows}
    print(json.dumps(stats, indent=2))
    if args.stats_out:
        Path(args.stats_out).write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
