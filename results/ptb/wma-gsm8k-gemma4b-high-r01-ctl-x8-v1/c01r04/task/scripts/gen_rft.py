#!/usr/bin/env python3
"""Rejection-sampling round: sample k solutions per problem from a checkpoint, keep the
ones whose final answer matches gold, and write them back out in the training format.

Problems come from the gsm8k TRAIN split and from OpenMathInstruct-2's augmented_gsm8k
problems (also train-seeded); the internal dev problems and the eval's 10 few-shot
problems are excluded, same as in build_data.py.
"""
import argparse, glob, json, os, random, re, sys
from collections import defaultdict
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_format import fewshot_prefix, fewshot_questions, render_prompt, render_target
from dev_eval import scorer_reads, norm

OMI2 = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
NUMERIC = re.compile(r"^\d+(\.\d+)?$")


def load_problems(n, seed, dev_problems):
    from datasets import load_dataset
    out = []
    tr = load_dataset("openai/gsm8k", "main", split="train")
    for rec in tr:
        q = rec["question"].strip()
        if q in dev_problems:
            continue
        out.append((q, rec["answer"].split("####")[-1].strip()))
    seen = {q for q, _ in out}
    for fp in sorted(glob.glob(OMI2)):
        if len(out) >= n:
            break
        df = pq.read_table(fp, columns=["problem", "expected_answer", "problem_source"]).to_pandas()
        df = df[df.problem_source == "augmented_gsm8k"]
        for q, a in zip(df["problem"].tolist(), df["expected_answer"].tolist()):
            q = q.strip()
            a = str(a).strip()
            if q in seen or q in dev_problems or not NUMERIC.match(a):
                continue
            seen.add(q)
            out.append((q, a))
            if len(out) >= n:
                break
    random.Random(seed).shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--n-problems", type=int, default=40000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--min-pass", type=float, default=0.01, help="keep problems whose pass@k rate is >= this")
    ap.add_argument("--max-pass", type=float, default=0.99, help="...and <= this; the frontier the model is still learning")
    ap.add_argument("--easy-keep-frac", type=float, default=0.15, help="fraction of fully-solved problems kept anyway")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev_problems = {json.loads(l)["question"].strip()
                    for l in open("/home/ben/task/data/dev_gsm8ktrain.jsonl")}
    dev_problems |= set(fewshot_questions())
    probs = load_problems(a.n_problems, a.seed, dev_problems)
    print(f"{len(probs)} problems", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_frac, max_model_len=1536,
              dtype="bfloat16", disable_log_stats=True)
    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p, n=a.k,
                        max_tokens=a.max_tokens, seed=a.seed, stop_token_ids=[1, 106])
    prompts = ["<bos>" + render_prompt(q, None) for q, _ in probs]
    outs = llm.generate(prompts, sp)

    rng = random.Random(a.seed)
    fs_prefix = fewshot_prefix()
    n_solved = n_kept_samples = n_unsolved = n_easy_dropped = 0
    rows = []                                   # (problem, solution_body, gold)
    for (q, gold), o in zip(probs, outs):
        good, seen = [], set()
        for c in o.outputs:
            if c.finish_reason != "stop":
                continue
            t = c.text.strip()
            if t.count("ANSWER: ") != 1:
                continue
            if norm(scorer_reads(t)) != norm(gold):
                continue
            body = t.rsplit("ANSWER: ", 1)[0].strip()
            if len(body) < 40:
                continue
            key = re.sub(r"\s+", " ", body)
            if key in seen:
                continue
            seen.add(key)
            good.append(body)
        if not good:
            n_unsolved += 1
            continue
        n_solved += 1
        rate = len(good) / a.k
        if rate > a.max_pass and rng.random() > a.easy_keep_frac:
            n_easy_dropped += 1
            continue                            # already reliably solved: little to learn
        if rate < a.min_pass:
            continue
        good.sort(key=len)                      # prefer the concise correct solutions
        for body in good[: a.keep_per_problem]:
            rows.append((q, body, gold))
            n_kept_samples += 1

    rng.shuffle(rows)
    n_fs = int(len(rows) * a.fewshot_frac)
    flags = [True] * n_fs + [False] * (len(rows) - n_fs)
    rng.shuffle(flags)
    with open(a.out, "w") as f:
        for (q, body, gold), use_fs in zip(rows, flags):
            f.write(json.dumps({
                "prompt": render_prompt(q, fs_prefix if use_fs else None),
                "completion": render_target(body, gold),
                "src": "rft", "fewshot": use_fs}) + "\n")
    stats = {"model": a.model, "n_problems": len(probs), "k": a.k,
             "pass_at_k_problems": n_solved / len(probs),
             "kept_rows": len(rows), "kept_per_solved": n_kept_samples / max(1, n_solved),
             "unsolved_problems": n_unsolved, "easy_dropped": n_easy_dropped,
             "min_pass": a.min_pass, "max_pass": a.max_pass, "easy_keep_frac": a.easy_keep_frac,
             "temperature": a.temperature, "top_p": a.top_p, "fewshot_rows": n_fs}
    print(json.dumps(stats, indent=1))
    if a.stats_out:
        json.dump(stats, open(a.stats_out, "w"), indent=1)


if __name__ == "__main__":
    main()
