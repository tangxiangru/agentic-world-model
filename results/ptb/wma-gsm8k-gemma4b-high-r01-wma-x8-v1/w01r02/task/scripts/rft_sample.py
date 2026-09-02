#!/usr/bin/env python3
"""Rejection-sampling generation: sample k solutions per training problem from a
checkpoint, keep the ones whose extracted answer matches gold, write an SFT file.

Prompt pool = gsm8k TRAIN problems and OpenMathInstruct-2 augmented_gsm8k
problems (both already in data/*.jsonl, probe300 excluded). The benchmark test
split is never used.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fmt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", default="data/sft_v2.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-problems", type=int, default=30000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--mix-file", type=str, default="")
    ap.add_argument("--mix-n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rng = random.Random(args.seed)
    seen, pool = set(), []
    for line in open(args.pool):
        r = json.loads(line)
        if r["question"] in seen:
            continue
        seen.add(r["question"])
        pool.append({"question": r["question"], "answer": r["answer"]})
    rng.shuffle(pool)
    pool = pool[: args.n_problems]
    print(f"pool {len(pool)} problems, k={args.k}", flush=True)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=1024, dtype="bfloat16")
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate([fmt.render_prompt(p["question"], None) for p in pool], sp)

    # dump raw generations first: a bug in post-processing must not cost the GPU pass
    raw_path = args.out + ".raw.jsonl"
    with open(raw_path, "w") as f:
        for p, o in zip(pool, outs):
            f.write(json.dumps({"question": p["question"], "answer": p["answer"],
                                "samples": [{"text": c.text, "fr": c.finish_reason} for c in o.outputs]}) + "\n")
    print("raw generations ->", raw_path, flush=True)

    fewshot_sys = open("data/fewshot_system.txt").read()
    import re

    shots = re.split(r"\n\n(?=[^\n]*\n\nReasoning:)", fewshot_sys)

    rows, n_solved, n_kept = [], 0, 0
    for p, o in zip(pool, outs):
        good = []
        for c in o.outputs:
            t = c.text.strip()
            if c.finish_reason == "length" or not t:
                continue
            if t.count("ANSWER:") != 1:
                continue
            if not fmt.grade(t, p["answer"]):
                continue
            good.append(t)
        if not good:
            continue
        n_solved += 1
        # frontier weighting: keep more paths for problems the model only sometimes solves
        keep = args.keep_per_problem if len(good) <= args.k // 2 else 1
        good = sorted(set(good), key=len)[:keep]
        for t in good:
            body = t.split("ANSWER:")[0].rstrip()
            rows.append({"question": p["question"], "target_body": body, "answer": p["answer"]})
            n_kept += 1

    # regularisation mixture: a slice of the original supervised file, verbatim
    mix = []
    if args.mix_file and args.mix_n > 0:
        allmix = [json.loads(l) for l in open(args.mix_file)]
        rng.shuffle(allmix)
        mix = allmix[: args.mix_n]
    print(f"rft rows {len(rows)}, mixed-in supervised rows {len(mix)}", flush=True)

    rng.shuffle(rows)
    n_fs = int(len(rows) * args.fewshot_frac)
    out = []
    for i, r in enumerate(rows):
        if i < n_fs:
            k = rng.choices([1, 2, 3, 4, 10], [0.30, 0.25, 0.15, 0.15, 0.15])[0]
            system = fewshot_sys if k >= 10 else "\n\n".join(rng.sample(shots, k))
        else:
            system = None
        comp = fmt.render_target(r["target_body"], r["answer"])
        assert fmt.grade(comp[: -len(fmt.END)], r["answer"])
        out.append({"prompt": fmt.render_prompt(r["question"], system), "completion": comp,
                    "question": r["question"], "answer": r["answer"], "src": "rft",
                    "nshot": (k if i < n_fs else 0)})
    out.extend(mix)
    rng.shuffle(out)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"problems": len(pool), "solved": n_solved,
                      "solve_rate": n_solved / len(pool), "rft_rows": len(rows), "rows": len(out)}, indent=2))


if __name__ == "__main__":
    main()
