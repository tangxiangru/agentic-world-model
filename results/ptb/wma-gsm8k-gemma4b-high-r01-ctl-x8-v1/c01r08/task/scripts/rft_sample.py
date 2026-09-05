"""Rejection-sampling fine-tuning data: sample the current model on training
problems, keep only solutions whose final answer is right, emit them in the
grader's rendered format.

Problems come from GSM8K *train* (minus the 300 held-out probe items and the
grader's 10 few-shot exemplars) and from OpenMathInstruct-2's gsm8k-flavoured
rows. No benchmark test item is used.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grader_format as gf  # noqa: E402

NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "")
    if not NUM_RE.match(a):
        return None
    if "." in a:
        f = float(a)
        if f.is_integer():
            a = str(int(f))
    return a


def collect_problems(n_gsm_train: int, n_omi: int, seed: int):
    from datasets import load_dataset

    rng = random.Random(seed)
    with open("data/dev_train300.jsonl") as f:
        dev_q = {json.loads(l)["question"].strip() for l in f}
    fewshot_q = {r["question"].strip() for r in
                 load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42).select(range(10))}

    probs = []
    tr = load_dataset("openai/gsm8k", "main", split="train")
    for r in tr:
        q = r["question"].strip()
        if q in dev_q or q in fewshot_q:
            continue
        a = norm_answer(r["answer"].split("####")[-1])
        if a:
            probs.append((q, a, "gsm8k_train"))
    rng.shuffle(probs)
    probs = probs[:n_gsm_train]

    if n_omi > 0:
        omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
        omi = omi.filter(lambda r: r["problem_source"] in ("gsm8k", "augmented_gsm8k"), num_proc=16)
        idx = list(range(len(omi)))
        rng.shuffle(idx)
        seen = {q for q, _, _ in probs}
        got = 0
        for i in idx:
            r = omi[i]
            q = r["problem"].strip()
            if q in seen or q in dev_q or q in fewshot_q:
                continue
            a = norm_answer(r["expected_answer"])
            if not a:
                continue
            seen.add(q)
            probs.append((q, a, "omi"))
            got += 1
            if got >= n_omi:
                break
    rng.shuffle(probs)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm-train", type=int, default=7163)
    ap.add_argument("--n-omi", type=int, default=25000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", default=None)
    args = ap.parse_args()

    probs = collect_problems(args.n_gsm_train, args.n_omi, args.seed)
    print(f"[rft] {len(probs)} problems x k={args.k}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=1536,
              dtype="bfloat16", seed=args.seed, enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed)
    prompts = [gf.render_prompt(q, None) for q, _, _ in probs]
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    kept, n_corr, n_tot, solved = [], 0, 0, 0
    per_source = {}
    for (q, gold, src), o in zip(probs, outs):
        cands = []
        for c in o.outputs:
            t = c.text
            n_tot += 1
            if gf.STOP_TOKEN in t:
                t = t.split(gf.STOP_TOKEN)[0]
            elif c.finish_reason != "stop":
                continue
            t = t.strip()
            if t.count(gf.ANSWER_MARKER) != 1:
                continue
            if not gf.score_completion(t, gold):
                continue
            n_corr += 1
            cands.append(t)
        s = per_source.setdefault(src, [0, 0])
        s[1] += 1
        if not cands:
            continue
        solved += 1
        s[0] += 1
        # keep more solutions for problems the model finds hard: a problem it
        # already gets right every time carries little new signal.
        n_keep = 1 if len(cands) == args.k else args.keep_per_problem
        uniq = sorted(set(cands), key=len)[:n_keep]
        for t in uniq:
            kept.append({"prompt": gf.render_prompt(q, None),
                         "completion": gf.render_target(t),
                         "fewshot": False, "source": src, "gold": gold})

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    stats = {"problems": len(probs), "samples": n_tot, "correct_samples": n_corr,
             "pass_rate": n_corr / max(1, n_tot), "problems_solved": solved,
             "solve_rate": solved / max(1, len(probs)), "rows_written": len(kept),
             "k": args.k, "temperature": args.temperature,
             "per_source": {k: {"solved": v[0], "n": v[1], "rate": v[0] / max(1, v[1])}
                            for k, v in per_source.items()}}
    print(json.dumps(stats, indent=1))
    if args.stats:
        json.dump(stats, open(args.stats, "w"), indent=1)


if __name__ == "__main__":
    main()
