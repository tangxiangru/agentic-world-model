#!/usr/bin/env python3
"""Sample k solutions per training question from an SFT checkpoint, keep the ones
whose final ANSWER matches the reference, and write them as new SFT rows.

Questions come from GSM8K *train* and from OpenMathInstruct-2's GSM8K-family
problems (both train-derived). No benchmark test item is read here.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from build_sft_data import (
    GSM8K_TRAIN,
    MATH_PROMPT_TEMPLATE,
    OMI2_GLOB,
    STOP_TOKEN,
    norm_answer,
)

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")


def gather_questions(n_omi2: int, seed: int, exclude: set[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    exclude = exclude or set()
    qs = []
    f = sorted(glob.glob(GSM8K_TRAIN))[0]
    t = pq.read_table(f)
    for q, a in zip(t.column("question").to_pylist(), t.column("answer").to_pylist()):
        gold = norm_answer(a.rpartition("####")[2])
        if gold is not None:
            qs.append({"question": q.strip(), "gold": gold, "src": "gsm8k:train"})
    seen = {r["question"] for r in qs} | exclude
    pool = []
    for f in sorted(glob.glob(OMI2_GLOB)):
        t = pq.read_table(f, columns=["problem", "expected_answer", "problem_source"])
        t = t.filter(pc.is_in(t.column("problem_source"), value_set=pa.array(["augmented_gsm8k"])))
        for p, a in zip(t.column("problem").to_pylist(), t.column("expected_answer").to_pylist()):
            gold = norm_answer(a)
            if gold is None or p.strip() in seen:
                continue
            seen.add(p.strip())
            pool.append({"question": p.strip(), "gold": gold, "src": "omi2:augmented_gsm8k"})
        del t
    rng.shuffle(pool)
    qs += pool[:n_omi2]
    rng.shuffle(qs)
    return qs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--n-omi2", type=int, default=20000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--hard-bonus", type=int, default=4,
                    help="keep up to this many correct samples for problems solved by <=1/k samples")
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-questions", default=None,
                    help="jsonl whose 'question' field lists OMI2 questions already sampled in an earlier round")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    exclude = set()
    if args.exclude_questions:
        exclude = {json.loads(l)["question"] for l in open(args.exclude_questions)}
        print(f"[gen] excluding {len(exclude)} questions sampled in an earlier round")
    qs = gather_questions(args.n_omi2, args.seed, exclude)
    print(f"[gen] {len(qs)} questions x k={args.k}")

    template = open("/home/ben/task/templates/gemma3.jinja").read()
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=4096,
              dtype="bfloat16", seed=args.seed)
    tok = llm.get_tokenizer()

    prompts = []
    for q in qs:
        user = MATH_PROMPT_TEMPLATE.format(prompt=q["question"])
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": user}], chat_template=template,
            tokenize=False, add_generation_prompt=True))
    print("[gen] example prompt:\n" + prompts[0][:300] + "\n...")

    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    rows, n_correct, n_total = [], 0, 0
    solved_hist = defaultdict(int)
    for q, prompt, out in zip(qs, prompts, outs):
        good = []
        for cand in out.outputs:
            n_total += 1
            text = cand.text.strip()
            m = ANS_RE.search(text)
            if not m:
                continue
            got = norm_answer(m.group(1))
            if got is None or got != q["gold"]:
                continue
            n_correct += 1
            good.append(text)
        solved_hist[len(good)] += 1
        if not good:
            continue
        # dedup, then keep more samples for the problems the model finds hard
        uniq = list(dict.fromkeys(good))
        rng.shuffle(uniq)
        cap = args.hard_bonus if len(good) <= 1 else args.max_per_problem
        for text in uniq[:cap]:
            rows.append({
                "prompt": MATH_PROMPT_TEMPLATE.format(prompt=q["question"]),
                "completion": text + STOP_TOKEN,
                "completion_body": text,
                "answer": q["gold"],
                "question": q["question"],
                "source": f"rft:{q['src']}",
            })

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    stats = {
        "questions": len(qs), "k": args.k,
        "samples": n_total, "correct_samples": n_correct,
        "pass_rate": n_correct / max(n_total, 1),
        "solved_histogram": {str(k): v for k, v in sorted(solved_hist.items())},
        "solve_any_rate": 1 - solved_hist[0] / max(len(qs), 1),
        "rows_written": len(rows), "out": args.out,
    }
    print(json.dumps(stats, indent=2))
    if args.stats_out:
        json.dump(stats, open(args.stats_out, "w"), indent=2)


if __name__ == "__main__":
    main()
