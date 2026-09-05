#!/usr/bin/env python3
"""Rejection-sampling fine-tuning data generation.

Samples k completions per training question from the current policy with vLLM,
keeps the ones whose final "ANSWER:" matches the gold answer, dedupes, and
writes a new SFT jsonl.
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict

from datasets import load_from_disk

from prep_sft import render_prompt, norm_answer

ANS_RE = re.compile(r"ANSWER:\s*([^\n]*)")


def extract(text: str):
    m = ANS_RE.findall(text)
    if not m:
        return None
    return norm_answer(m[-1])


def sig(sol: str) -> str:
    """Signature used for de-duplicating reasoning chains: the ordered list of
    numbers that appear in the solution."""
    return ",".join(re.findall(r"-?\d+\.?\d*", sol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--keep-per-problem", type=int, default=4)
    ap.add_argument("--n-augmented", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gsm = load_from_disk("data/gsm8k_main")["train"]
    items = []  # (question, gold)
    for r in gsm:
        gold = norm_answer(r["answer"].split("####")[-1].strip())
        if gold is not None:
            items.append((r["question"].strip(), gold))

    if args.n_augmented > 0:
        omi = load_from_disk("data/omi2_gsm")
        seen = set()
        pool = []
        for r in omi:
            if r["problem_source"] != "augmented_gsm8k":
                continue
            p = r["problem"].strip()
            if p in seen:
                continue
            seen.add(p)
            a = norm_answer(r["expected_answer"])
            if a is not None:
                pool.append((p, a))
        rng.shuffle(pool)
        items += pool[: args.n_augmented]

    print(f"{len(items)} problems x {args.n} samples")

    # few-shot pool (GSM8K train reference solutions, same rendering as inspect)
    fewshot_pool = []
    for r in gsm:
        ans = r["answer"].split("####")
        target = ans.pop().strip()
        fewshot_pool.append(
            f"{r['question']}\n\nReasoning:\n{'####'.join(ans).strip()}\n\nANSWER: {target}"
        )

    prompts, meta = [], []
    for q, gold in items:
        system = None
        if rng.random() < args.fewshot_frac:
            system = "\n\n".join(rng.sample(fewshot_pool, rng.choice([1, 2, 3])))
        prompts.append(render_prompt(q, system))
        meta.append((q, gold, system))

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=2048,
        dtype="bfloat16",
        enable_prefix_caching=True,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.n,
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<end_of_turn>"],
        seed=None,
    )
    outs = llm.generate(prompts, sp)

    kept = 0
    n_solved = 0
    per_problem_rate = []
    with open(args.out, "w") as f:
        for out, (q, gold, system) in zip(outs, meta):
            good = []
            n_ok = 0
            for c in out.outputs:
                txt = c.text.strip()
                if extract(txt) != gold:
                    continue
                n_ok += 1
                if c.finish_reason != "stop" or len(txt) < 60:
                    continue
                good.append(txt)
            rate = n_ok / max(1, len(out.outputs))
            per_problem_rate.append(rate)
            if n_ok:
                n_solved += 1
            # up-weight harder problems: keep fewer chains for ones the policy
            # already always gets right
            budget = args.keep_per_problem if rate < 0.9 else max(1, args.keep_per_problem // 2)
            # dedupe by numeric signature over a random order (no length bias)
            rng.shuffle(good)
            seen_sig, chosen = set(), []
            for g in good:
                s = sig(g)
                if s in seen_sig:
                    continue
                seen_sig.add(s)
                chosen.append(g)
                if len(chosen) >= budget:
                    break
            for g in chosen:
                f.write(json.dumps({
                    "prompt_text": render_prompt(q, system),
                    "completion_text": g + "<end_of_turn>",
                    "question": q,
                    "answer": gold,
                }) + "\n")
                kept += 1

    print(f"solved {n_solved}/{len(items)} problems; kept {kept} solutions")
    print(f"mean pass rate {sum(per_problem_rate)/len(per_problem_rate):.3f}")
    if args.stats_out:
        json.dump({"n_solved": n_solved, "n_items": len(items), "kept": kept,
                   "mean_pass_rate": sum(per_problem_rate) / len(per_problem_rate)},
                  open(args.stats_out, "w"), indent=2)


if __name__ == "__main__":
    main()
