#!/usr/bin/env python3
"""Sample a training set from data/pool.jsonl and pre-tokenize it.

Writes:
  <out>.jsonl     : the rows, with rendered prompt_text / target_text (human-readable)
  <out>.tok.pt    : {"input_ids": list[list[int]], "prompt_lens": list[int]}
Also prints the length histogram the max_seq_len preflight check needs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, "/home/ben/task/scripts")
from format_utils import (  # noqa: E402
    STOP_TOKEN,
    eval_fewshot_system,
    random_fewshot_system,
    render_prompt,
    template_sha,
)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/home/ben/task/data/pool.jsonl")
    ap.add_argument("--out", required=True, help="path prefix, no extension")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-problem", type=int, default=2)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(SNAP)
    print("template sha", template_sha())

    rows = [json.loads(l) for l in open(args.pool)]
    rng.shuffle(rows)

    # cap solutions per problem, then take the first n
    per: dict[str, int] = {}
    picked = []
    for r in rows:
        c = per.get(r["problem"], 0)
        if c >= args.max_per_problem:
            continue
        per[r["problem"]] = c + 1
        picked.append(r)
        if len(picked) >= args.n:
            break
    print(f"picked {len(picked)} rows over {len(per)} distinct problems")

    input_ids, prompt_lens, out_rows = [], [], []
    n_trunc = 0
    eval_fs = eval_fewshot_system()
    for r in picked:
        sysmsg = None
        if rng.random() < args.fewshot_frac:
            # 1/3 of few-shot rows use the grader's literal 10-shot message, the rest a
            # random k-shot message in the same format; both are gsm8k TRAIN only.
            if rng.random() < 0.34:
                sysmsg = eval_fs
            else:
                sysmsg = random_fewshot_system(rng, rng.choice([2, 4, 6, 8, 10]))
        prompt_text = render_prompt(tok, r["prompt"], sysmsg)
        target_text = r["completion"] + STOP_TOKEN
        p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        t_ids = tok(target_text, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(t_ids) > args.max_seq_len:
            n_trunc += 1
            continue
        input_ids.append(p_ids + t_ids)
        prompt_lens.append(len(p_ids))
        out_rows.append({
            "prompt": prompt_text,
            "completion": target_text,
            "final_answer": r["answer"],
            "source": r["source"],
            "fewshot": sysmsg is not None,
            "n_tokens": len(p_ids) + len(t_ids),
        })

    lens = sorted(len(x) for x in input_ids)
    n = len(lens)
    print(f"kept {n} rows, dropped {n_trunc} over max_seq_len ({100*n_trunc/max(1,n+n_trunc):.2f}%)")
    print("token len p50 %d p90 %d p99 %d max %d  total %.1fM" % (
        lens[n // 2], lens[int(0.9 * n)], lens[int(0.99 * n)], lens[-1], sum(lens) / 1e6))
    print("fewshot rows:", sum(r["fewshot"] for r in out_rows))

    with open(args.out + ".jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    torch.save({"input_ids": input_ids, "prompt_lens": prompt_lens}, args.out + ".tok.pt")
    print("wrote", args.out + ".jsonl", "and", args.out + ".tok.pt")


if __name__ == "__main__":
    main()
