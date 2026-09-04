"""Rejection-sampling fine-tuning data: sample k solutions per GSM8K-train
question from a checkpoint, keep the ones whose final number is correct.

Uses the same rendered prompt as training/grading (scripts/fmt.py) and the same
answer convention (last integer in the response is the answer).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

INT_RE = re.compile(r"-?\d[\d,]*")


def final_number(text: str) -> str | None:
    m = INT_RE.findall(text.replace("$", ""))
    if not m:
        return None
    return m[-1].replace(",", "").lstrip("+")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-keep", type=int, default=2)
    ap.add_argument("--n-questions", type=int, default=0, help="0 = all non-holdout train questions")
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    g = load_dataset("openai/gsm8k", "main")["train"]
    hold = set(json.load(open("/home/ben/task/data/holdout_idx.json")))
    items = []
    for i in range(len(g)):
        if i in hold:
            continue
        gold = g[i]["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
        items.append({"idx": i, "question": g[i]["question"].strip(), "gold": gold})
    if args.n_questions:
        items = items[: args.n_questions]
    print(f"[rft] {len(items)} questions x k={args.k}", flush=True)

    prompts = [fmt.render_prompt(it["question"], fewshot=False) for it in items]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=1024,
        dtype="bfloat16",
        seed=args.seed,
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temp,
        top_p=0.95,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    rows = []
    n_correct_q = 0
    n_any = 0
    per_q_correct = []
    for it, o in zip(items, outs):
        cands = []
        for c in o.outputs:
            t = c.text.strip()
            if final_number(t) == it["gold"] and t.count("ANSWER:") == 1:
                cands.append(t)
        per_q_correct.append(len(cands))
        n_any += 1
        if not cands:
            continue
        n_correct_q += 1
        # dedup, prefer shorter (less rambling) solutions
        seen = set()
        uniq = []
        for t in sorted(cands, key=len):
            key = re.sub(r"\s+", " ", t)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(t)
        for t in uniq[: args.max_keep]:
            rows.append({"question": it["question"], "answer": t, "src": "rft", "idx": it["idx"]})

    rng.shuffle(rows)
    n_few = int(round(args.fewshot_frac * len(rows)))
    with open(args.out, "w") as f:
        for k, r in enumerate(rows):
            few = k < n_few
            f.write(
                json.dumps(
                    {
                        "question": r["question"],
                        "answer": r["answer"],
                        "prompt": fmt.render_prompt(r["question"], fewshot=few),
                        "completion": fmt.render_target(r["answer"]),
                        "fewshot": few,
                        "src": "rft",
                        "idx": r["idx"],
                    }
                )
                + "\n"
            )
    stats = {
        "questions": n_any,
        "questions_with_a_correct_sample": n_correct_q,
        "pass_at_k": n_correct_q / max(1, n_any),
        "mean_correct_per_question": sum(per_q_correct) / max(1, len(per_q_correct)),
        "k": args.k,
        "temp": args.temp,
        "rows_written": len(rows),
        "unsolved_idx": [it["idx"] for it, c in zip(items, per_q_correct) if c == 0][:2000],
    }
    if args.stats:
        json.dump(stats, open(args.stats, "w"), indent=2)
    print({k: v for k, v in stats.items() if k != "unsolved_idx"}, flush=True)


if __name__ == "__main__":
    main()
