#!/usr/bin/env python3
"""Rejection sampling: draw k solutions per question from a checkpoint with vLLM,
keep the ones whose final ANSWER matches gold, dedup, and write an SFT file in the
same schema as data/train_sft.jsonl.

Prompts are rendered with scripts/prompting.py (the grader's template) so the
samples are on-policy for the evaluation distribution.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from prompting import render_prompt  # noqa: E402

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def norm(x: str) -> str | None:
    x = x.strip().replace(",", "")
    if not x:
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(v)


def final_answer(text: str) -> str | None:
    """What match(numeric=True, location='end') would read: the last number."""
    ms = NUM.findall(text)
    return norm(ms[-1]) if ms else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with question + gold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--out-unsolved", default=None,
                    help="dump questions with zero correct samples for a second, deeper pass")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        rows = rows[: args.limit]
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    # tokenize here with add_special_tokens=False: render_prompt already emits the
    # template's <bos>, and LLM.generate() on a raw string would prepend a second
    # one, which is not what the grader's chat endpoint sends.
    _tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [TokensPrompt(prompt_token_ids=_tok(render_prompt(r["question"], None),
                                                  add_special_tokens=False)["input_ids"])
               for r in rows]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, dtype="bfloat16", seed=args.seed,
              enable_prefix_caching=True)
    # vLLM's offline generate() with an explicit SamplingParams does not inherit
    # generation_config's eos list, so name both terminators explicitly; without
    # this the model writes <end_of_turn><start_of_turn>model and keeps going.
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    kept, per_q, unsolved = [], defaultdict(list), []
    n_gen = n_ok = 0
    solved = 0
    for r, o in zip(rows, outs):
        gold = norm(str(r["gold"]))
        good = []
        for c in o.outputs:
            n_gen += 1
            t = c.text
            if c.finish_reason != "stop":
                continue
            if t.count("ANSWER:") != 1:
                continue
            if final_answer(t) != gold:
                continue
            body = t.strip()
            n_ok += 1
            good.append(body)
        if good:
            solved += 1
        else:
            unsolved.append(r)
        # keep at most max_per_question distinct correct chains, chosen at random
        # so the mixture is not biased toward degenerate one-line "the answer is X"
        uniq = [u for u in dict.fromkeys(good) if len(u) >= 80]
        rng.shuffle(uniq)
        for body in uniq[: args.max_per_question]:
            per_q[r["question"]].append(body)

    for q, bodies in per_q.items():
        for b in bodies:
            kept.append({"question": q, "answer": b,
                         "target": b + "<end_of_turn>", "source": "rft_self"})
    rng.shuffle(kept)
    with open(args.out, "w") as f:
        for k in kept:
            f.write(json.dumps(k) + "\n")

    if args.out_unsolved:
        with open(args.out_unsolved, "w") as f:
            for r in unsolved:
                f.write(json.dumps(r) + "\n")

    stats = {"questions": len(rows), "samples_generated": n_gen,
             "samples_correct": n_ok, "pass_rate": n_ok / max(n_gen, 1),
             "questions_with_at_least_one": solved,
             "solve_rate": solved / max(len(rows), 1), "rows_written": len(kept)}
    print(json.dumps(stats, indent=2))
    if args.stats_out:
        json.dump(stats, open(args.stats_out, "w"), indent=2)


if __name__ == "__main__":
    main()
