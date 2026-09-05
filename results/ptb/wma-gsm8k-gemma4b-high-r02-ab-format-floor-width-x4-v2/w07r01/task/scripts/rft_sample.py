"""Rejection-sampling fine-tuning: sample k solutions per GSM8K *train* question from a
checkpoint, keep the ones whose 'ANSWER: N' matches gold, dedup, write an SFT jsonl.

Prompts are rendered with the same templates/gemma3.jinja the grader uses, so the
samples are drawn from exactly the distribution the model is graded in.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SNAPSHOT = os.environ["PTB_BASE_MODEL_SNAPSHOT"]
NUM = re.compile(r"-?[\d,]*\.?\d+")


def extract(text: str) -> str | None:
    """Mirror inspect's match(numeric=True, location='end'): last numeric token."""
    toks = re.split(r"\s+", text.strip().replace(",", "").replace("$", ""))
    for t in reversed(toks):
        t2 = t.rstrip(".").rstrip("*")
        if t2.replace(".", "").replace("-", "").isnumeric():
            return fmt.normalize_answer(t2)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--n-questions", type=int, default=None)
    ap.add_argument("--holdout", default="data/dev300.jsonl",
                    help="jsonl of private-dev questions that must never be trained on")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    tok.chat_template = fmt.load_template()

    hold = set()
    if args.holdout and os.path.exists(args.holdout):
        hold = {json.loads(l)["question"] for l in open(args.holdout)}
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    qs = []
    n_held = 0
    for r in gsm:
        if r["question"] in hold:
            n_held += 1
            continue
        a = fmt.normalize_answer(r["answer"].split("####")[-1])
        if a is not None:
            qs.append((r["question"], a))
    print(f"[rft] excluded {n_held} private-dev holdout questions")
    if args.n_questions:
        qs = qs[: args.n_questions]
    print(f"[rft] {len(qs)} train questions, k={args.k}")

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": fmt.user_prompt(q)}],
            tokenize=False, add_generation_prompt=True,
        )
        for q, _ in qs
    ]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, dtype="bfloat16", seed=args.seed)
    # No per-request seed: it makes vLLM build one RNG per sequence and slows the batch.
    # The engine-level seed above already pins the run.
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop=[fmt.STOP_TOKEN])
    outs = llm.generate(prompts, sp)

    n_corr = n_any = 0
    kept = 0
    per_q_correct = []
    with open(args.out, "w") as f:
        for (q, gold), o in zip(qs, outs):
            texts, seen = [], set()
            n_ok = 0
            for c in o.outputs:
                t = c.text.strip()
                if extract(t) != gold:
                    continue
                n_ok += 1
                # one clean 'ANSWER: N' line at the end, nothing after it
                m = list(re.finditer(r"^ANSWER:\s*\S+\s*$", t, flags=re.M))
                if len(m) != 1 or not t.endswith(m[0].group(0).strip()):
                    continue
                key = re.sub(r"\s+", " ", t)[:400]
                if key in seen:
                    continue
                seen.add(key)
                texts.append(t)
            per_q_correct.append(n_ok)
            n_corr += n_ok
            n_any += bool(n_ok)
            # prefer the shortest correct solutions: less room for a lucky wrong chain
            texts.sort(key=len)
            for t in texts[: args.max_per_question]:
                prompt = tok.apply_chat_template(
                    [{"role": "user", "content": fmt.user_prompt(q)}],
                    tokenize=False, add_generation_prompt=True,
                )
                f.write(json.dumps({
                    "prompt": prompt,
                    "completion": t + fmt.STOP_TOKEN,
                    "answer": gold,
                    "src": "rft",
                    "question": q,
                }) + "\n")
                kept += 1

    stats = {
        "model": args.model,
        "holdout_excluded": n_held,
        "questions": len(qs),
        "k": args.k,
        "temp": args.temp,
        "pass_rate_per_sample": n_corr / (len(qs) * args.k),
        "pass_at_k": n_any / len(qs),
        "rows_written": kept,
        "questions_with_no_correct_sample": sum(1 for c in per_q_correct if c == 0),
    }
    print(json.dumps(stats, indent=2))
    if args.stats:
        with open(args.stats, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
