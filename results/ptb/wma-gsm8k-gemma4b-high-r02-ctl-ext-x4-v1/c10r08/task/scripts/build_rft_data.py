#!/usr/bin/env python3
"""Turn gen_vllm.py --k samples into a rejection-sampling fine-tuning corpus.

Keeps only completions whose graded answer (the grader's own last-numeric-token
rule) equals gold, dedups, caps per question, and re-emits them in the same row
shape as data/sft_v1.jsonl so scripts/train_sft.py can read either file.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402


def normal_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--mix-sft", default=None, help="jsonl to blend in")
    ap.add_argument("--n-mix", type=int, default=25000)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    stats = collections.Counter()
    n_q = 0
    solved = 0
    for line in open(args.samples):
        r = json.loads(line)
        n_q += 1
        gold = r["gold"]
        keep = []
        seen = set()
        for comp, ok, fin in zip(r["completions"], r["correct"], r["finish"]):
            stats["total"] += 1
            if not ok:
                stats["wrong_answer"] += 1
                continue
            if fin != "stop":
                stats["no_stop"] += 1           # ran to max_tokens: never train on a truncated chain
                continue
            body = comp.strip()
            lines = body.split("\n")
            if not lines[-1].startswith("ANSWER:"):
                stats["no_answer_line"] += 1
                continue
            if body.count("ANSWER:") != 1:
                stats["multi_marker"] += 1
                continue
            k = normal_key(body)
            if k in seen:
                stats["dup"] += 1
                continue
            seen.add(k)
            keep.append(body)
        if keep:
            solved += 1
        # shortest-first: the terse correct chain is the cleaner training signal
        keep.sort(key=len)
        for body in keep[: args.max_per_question]:
            reasoning = "\n".join(body.split("\n")[:-1]).strip()
            rows.append({"question": r["question"],
                         "target": fmt.build_target(reasoning, gold) + "<end_of_turn>",
                         "answer": fmt.normalise_number(gold),
                         "source": "rft_self"})
            stats["kept"] += 1

    print(f"questions {n_q}, at least one correct sample {solved} "
          f"({solved / max(1, n_q):.3f})", flush=True)
    print(dict(stats), flush=True)

    if args.mix_sft:
        pool = [json.loads(l) for l in open(args.mix_sft)]
        rng.shuffle(pool)
        for r in pool[: args.n_mix]:
            rows.append({"question": r["question"], "target": r["target"],
                         "answer": r["answer"], "source": "sft_replay"})
        print(f"blended {min(args.n_mix, len(pool))} replay rows from {args.mix_sft}",
              flush=True)

    # few-shot prefixes, same recipe as build_sft_data.py
    fewshot_pool = []
    dev_ids = set(json.load(open("/home/ben/task/data/train_exclude_ids.json")))
    from datasets import load_dataset
    g = load_dataset("openai/gsm8k", "main")["train"]
    for i, rr in enumerate(g):
        if i in dev_ids:
            continue
        body, final = rr["answer"].split("####")
        fewshot_pool.append((rr["question"].strip(),
                             re.sub(r"<<[^>]*>>", "", body).strip(),
                             fmt.normalise_number(final)))

    rng.shuffle(rows)
    n_few = int(len(rows) * args.fewshot_frac)
    for i, row in enumerate(rows):
        if i < n_few:
            kshot = rng.choice([1, 2, 3, 4, 8])
            shots = rng.sample(fewshot_pool, kshot)
            row["system"] = "\n\n".join(fmt.fewshot_block(q, b, a) for q, b, a in shots)
            row["nshot"] = kshot
        else:
            row["system"] = None
            row["nshot"] = 0
    rng.shuffle(rows)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}", flush=True)
    print(collections.Counter(r["source"] for r in rows), flush=True)

    with open(args.out.replace(".jsonl", "_contam.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")


if __name__ == "__main__":
    main()
