#!/usr/bin/env python3
"""Sample solutions from a trained checkpoint and keep the ones that are right.

On-policy rejection-sampling data: the model's own phrasing, filtered by the
gold answer of the *training* problem (GSM8K train-derived, never test).

    python gen_rft.py --model ckpts/exp-02/final --problems data/sft_train2.jsonl \
        --n-problems 24000 --k 4 --out data/rft.jsonl
"""
import argparse
import json
import math
import random
import re
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

TEMPLATE = Path("templates/gemma3.jinja").read_text()
STOP_TOKEN = "<end_of_turn>"
ANS_RE = re.compile(r"ANSWER:\s*(-?[\d,]+(?:\.\d+)?)\s*$")


def final_answer(text: str) -> str | None:
    """Normalise the final ANSWER: number, or None if there is not exactly one.

    Must never raise: it runs after ~1 h of generation, and a crash here throws
    all of it away (this happened once, on a sample whose answer overflowed to
    inf). Anything unparseable is simply not a match.
    """
    m = ANS_RE.search(text.strip())
    if not m:
        return None
    v = m.group(1).replace(",", "")
    if len(v) > 18:
        return None
    try:
        f = float(v)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(f):
        return None
    try:
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", default="data/sft_train2.jsonl")
    ap.add_argument("--n-problems", type=int, default=24000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default="data/rft.jsonl")
    ap.add_argument("--stats", default="analysis/rft_stats.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.problems)]
    # zero-shot rows only: the k-shot prefix is a training-time device, and the
    # question text is what we want to resample.
    rows = [r for r in rows if not r.get("system")]
    rng.shuffle(rows)
    rows = rows[: args.n_problems]
    print(f"[rft] {len(rows)} problems x k={args.k}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            chat_template=TEMPLATE,
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in rows
    ]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        enforce_eager=False,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
        stop=[STOP_TOKEN],
    )
    outs = llm.generate(prompts, sp)

    # persist the raw generations before parsing anything: a parser bug must
    # never be able to throw away an hour of GPU time again.
    raw_path = Path(args.out).with_suffix(".raw.jsonl")
    with open(raw_path, "w") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"prompt": r["prompt"], "answer": r["answer"],
                                "samples": [c.text for c in o.outputs]}) + "\n")
    print(f"[raw] wrote {raw_path}", flush=True)

    kept, n_correct, n_total = [], 0, 0
    solved = 0
    for r, o in zip(rows, outs):
        gold = r["answer"]
        good = []
        for c in o.outputs:
            n_total += 1
            text = c.text.strip()
            got = final_answer(text)
            if got is not None and got == gold:
                n_correct += 1
                good.append(text)
        if good:
            solved += 1
        # keep distinct correct solutions at random: picking the shortest would
        # bias the corpus toward terse chains, which is the opposite of what
        # helps multi-step arithmetic.
        uniq = sorted(set(good))
        rng.shuffle(uniq)
        good = [g for g in uniq if len(g) >= 80][: args.keep_per_problem]
        for g in good:
            kept.append(
                {
                    "system": None,
                    "prompt": r["prompt"],
                    "completion": g + STOP_TOKEN,
                    "answer": gold,
                }
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    stats = {
        "problems": len(rows),
        "k": args.k,
        "samples": n_total,
        "correct_samples": n_correct,
        "sample_accuracy": n_correct / max(n_total, 1),
        "problems_solved_at_least_once": solved,
        "pass_at_k": solved / max(len(rows), 1),
        "kept_rows": len(kept),
    }
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
