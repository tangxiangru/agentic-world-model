"""Rejection-sampling round: sample k solutions per problem from our own model,
keep the ones whose final answer the grader's own matcher accepts.

Problems come from GSM8K *train* and from OpenMathInstruct-2's gsm8k-sourced
augmentations that exp-02 did NOT train on, so this adds on-policy solutions to
unseen problems rather than re-reciting the SFT set. Answers are the dataset's
own expected_answer; no benchmark item is involved.

    python rft_sample.py --model ckpts/exp-02/final --n-problems 20000 --k 4
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re

import pyarrow.parquet as pq
from inspect_ai.scorer._common import match_str

import gsm_format as G

OMI2 = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
GSM8K_TRAIN = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"
NUM_RE = re.compile(r"^-?\d{1,12}(\.\d{1,4})?$")


def clean_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "")
    if not NUM_RE.match(a):
        return None
    return a[:-2] if a.endswith(".0") else a


def used_questions(path="data/sft_train.jsonl") -> set[str]:
    """Recover the question text out of the rendered prompts of the SFT file."""
    head = G.MATH_PROMPT_TEMPLATE.split("{prompt}")[0].strip()
    tail = G.MATH_PROMPT_TEMPLATE.split("{prompt}")[1].split("\n\n")[1]
    out = set()
    for line in open(path):
        p = json.loads(line)["prompt"]
        i = p.rindex(head) + len(head)
        j = p.index(tail, i)
        out.add(p[i:j].strip())
    return out


def problem_pool(n: int, seed: int, exclude: set[str]):
    pool, seen = [], set()
    for f in sorted(glob.glob(OMI2)):
        df = pq.read_table(f, columns=["problem", "expected_answer", "problem_source"]).to_pandas()
        df = df[df.problem_source.isin(["gsm8k", "augmented_gsm8k"])]
        for problem, ans, _ in df.itertuples(index=False):
            if problem in seen or problem in exclude:
                continue
            a = clean_answer(ans)
            if a is None:
                continue
            seen.add(problem)
            pool.append((problem, a))
        print(f"  pool {len(pool)} after {f.split('/')[-1]}", flush=True)
        if len(pool) >= n * 2:
            break
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-problems", type=int, default=20000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/rft_train.jsonl")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    args = ap.parse_args()

    exclude = used_questions()
    print("questions already trained on:", len(exclude), flush=True)
    pool = problem_pool(args.n_problems, args.seed, exclude)
    print("problems to sample:", len(pool), flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model, gpu_memory_utilization=args.gpu_util,
        max_model_len=1024, dtype="bfloat16", enforce_eager=False, seed=args.seed,
    )
    sp = SamplingParams(
        n=args.k, temperature=args.temp, top_p=args.top_p, top_k=64,
        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed,
    )
    prompts = [G.render_prompt(q) for q, _ in pool]
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    n_kept = n_solved = 0
    with open(args.out, "w") as f, open("data/rft_contam.jsonl", "w") as fc:
        for (question, gold), o in zip(pool, outs):
            good = []
            for c in o.outputs:
                body = c.text.strip()
                if body.count("ANSWER: ") != 1:
                    continue
                _, ok = match_str(body, gold, location="end", numeric=True)
                if ok:
                    good.append(body)
            if not good:
                continue
            n_solved += 1
            rng.shuffle(good)
            # prefer the shortest correct chains: less rambling, fewer chances to slip
            good.sort(key=len)
            for body in good[: args.keep_per_problem]:
                reasoning = body.rsplit("\n\nANSWER: ", 1)[0]
                f.write(json.dumps({
                    "prompt": G.render_prompt(question),
                    "completion": G.render_target(reasoning, gold),
                }) + "\n")
                fc.write(json.dumps({"question": question, "answer": body}) + "\n")
                n_kept += 1

    print(json.dumps({
        "problems": len(pool), "solved_at_least_once": n_solved,
        "pass_at_k": round(n_solved / max(1, len(pool)), 4),
        "rows_written": n_kept, "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
