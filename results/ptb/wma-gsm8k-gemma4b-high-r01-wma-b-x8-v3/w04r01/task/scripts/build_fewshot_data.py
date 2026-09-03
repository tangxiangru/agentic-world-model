"""Build SFT rows whose prompts carry a k-shot prefix, k drawn per row.

exp-02 diagnosis (analysis/probe_exp02_final.json): the model trained only on zero-shot
prompts answers correctly and stops (bad_format 0.5%, acc 0.76) but, behind the grader's
10-shot system prefix, continues the few-shot pattern instead of stopping (bad_format 79.5%,
acc 0.39).  The grader ALWAYS prepends 10 shots, so the training prompts must too.

Few-shot blocks are rendered exactly as inspect_evals.gsm8k.sample_to_fewshot does
("{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}", joined by "\n\n"), with the
<<calculator>> annotations left in, because that is what the grader shows the model.
Shots are drawn from openai/gsm8k split=train only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402
from build_sft_data import CALC, clean_number, gsm8k_train_rows, omi2_rows  # noqa: E402

from datasets import load_dataset  # noqa: E402


def fewshot_pool():
    """(question, reasoning-with-<<>>, target) exactly as record_to_sample splits them."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    pool = []
    for r in ds:
        body, _, ans = r["answer"].rpartition("####")
        ans = ans.strip()
        if not ans:
            continue
        pool.append((r["question"], body.strip(), ans))
    return pool


def shot_block(q, reasoning, target) -> str:
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm8k", type=int, default=5000)
    ap.add_argument("--n-omi2", type=int, default=10000)
    ap.add_argument("--omi2", default="/home/ben/task/data/omi2_1M")
    ap.add_argument("--max-per-problem", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = fewshot_pool()

    rows = []
    g = list(gsm8k_train_rows())
    rng.shuffle(g)
    rows += g[: args.n_gsm8k]

    seen_q = {}
    o = []
    for q, body, ans, src in omi2_rows(args.omi2):
        if len(body) > 3000:
            continue
        if seen_q.get(q, 0) >= args.max_per_problem:
            continue
        seen_q[q] = seen_q.get(q, 0) + 1
        o.append((q, body, ans, src))
        if len(o) >= args.n_omi2 * 3:
            break
    rng.shuffle(o)
    rows += o[: args.n_omi2]
    rng.shuffle(rows)

    # k=10 is what the grader uses; the rest keeps the zero-shot ability exp-02 already has
    ks, ws = [0, 3, 6, 10], [0.30, 0.10, 0.15, 0.45]
    counts = {k: 0 for k in ks}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for q, body, ans, src in rows:
            k = rng.choices(ks, ws)[0]
            counts[k] += 1
            sysmsg = None
            if k:
                shots = rng.sample(pool, k + 2)
                shots = [s for s in shots if s[0] != q][:k]
                sysmsg = "\n\n".join(shot_block(*s) for s in shots)
            rec = {
                "prompt": fmt.render_prompt(q, fewshot_system=sysmsg),
                "completion": fmt.render_completion(body, ans),
                "source": f"{src}_k{k}",
                "question": q,
                "answer": ans,
                "k": k,
            }
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}; k distribution {counts}")


if __name__ == "__main__":
    main()
