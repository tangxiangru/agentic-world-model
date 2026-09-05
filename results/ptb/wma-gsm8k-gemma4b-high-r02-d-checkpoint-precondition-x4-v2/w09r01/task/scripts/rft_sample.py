#!/usr/bin/env python3
"""Rejection-sampling round: draw k solutions per GSM8K *train* problem from a
checkpoint, keep the ones that reach the reference answer, emit SFT rows.

Only the train split is touched. The test split never enters this file; the
output still goes through ../contamination_check.py before it is trained on.

Rows are rendered with the grader's own template (scripts/eval_format.py), so
the output file is drop-in for scripts/train_sft.py.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

from transformers import AutoTokenizer

import eval_format as ef

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def norm(s):
    s = s.strip().replace(",", "").replace("$", "")
    if not NUM.fullmatch(s):
        return None
    f = float(s)
    return str(int(f)) if f == int(f) else str(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--pool", default=None, help="jsonl of {question, answer}; default is the gsm8k train split")
    ap.add_argument("--n-problems", type=int, default=100000)
    ap.add_argument("--max-keep-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-every", type=int, default=8)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if args.pool:
        probs = [(r["question"], r["answer"]) for r in map(json.loads, open(args.pool))]
    else:
        probs = [(r["question"], norm(r["answer"].rsplit("####", 1)[1])) for r in ds]
    probs = [(q, a) for q, a in probs if a is not None][: args.n_problems]

    prompts = [ef.render(tok, q, None, None)[0] for q, _ in probs]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=2816,
              dtype="bfloat16", enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    kept = defaultdict(list)
    stats = defaultdict(int)
    solved = 0
    for (q, gold), o in zip(probs, outs):
        any_ok = False
        for c in o.outputs:
            stats["samples"] += 1
            txt = c.text.strip()
            if c.finish_reason != "stop":
                stats["drop_truncated"] += 1
                continue
            nums = NUM.findall(txt.replace(",", ""))
            if not nums or norm(nums[-1]) != gold:
                stats["drop_wrong"] += 1
                continue
            if txt.count(ef.ANSWER_MARKER) != 1 or not txt.split("\n")[-1].startswith(ef.ANSWER_MARKER):
                stats["drop_format"] += 1
                continue
            any_ok = True
            if txt not in kept[q] and len(kept[q]) < args.max_keep_per_problem:
                kept[q].append(txt)
        solved += any_ok
    stats["problems"] = len(probs)
    stats["problems_solved"] = solved
    stats["pass_at_k"] = round(solved / len(probs), 4)

    shots_pool = [(r["question"], *[x.strip() for x in r["answer"].rsplit("####", 1)]) for r in ds]
    rows = []
    for i, (q, ts) in enumerate(kept.items()):
        for t in ts:
            system = None
            n_shots = 0
            if args.fewshot_every and i % args.fewshot_every == 0:
                n_shots = rng.choice([2, 4, 10, 10])
                system = ef.fewshot_block(rng.sample(shots_pool, n_shots))
            prompt, full = ef.render(tok, q, system, t)
            completion = full[len(prompt):]
            completion = completion[: completion.rindex(ef.STOP_TOKEN) + len(ef.STOP_TOKEN)]
            rows.append({"prompt": prompt, "completion": completion, "n_shots": n_shots})
    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats["rows_written"] = len(rows)
    json.dump(dict(stats), open(args.stats_out, "w"), indent=2)
    print(json.dumps(dict(stats), indent=2))


if __name__ == "__main__":
    main()
