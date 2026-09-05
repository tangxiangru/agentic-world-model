#!/usr/bin/env python3
"""Rejection-sampling data generation from a trained checkpoint.

Samples k solutions per problem with vLLM using the grader's own prompt
rendering, keeps only the ones whose 'ANSWER: N' matches the reference answer,
dedups, and writes rows in the same {prompt, completion} shape build_data.py
emits so train_sft.py can consume them unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from transformers import AutoTokenizer
from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE

TASK = Path(__file__).resolve().parent
TEMPLATE = (TASK / "templates" / "gemma3.jinja").read_text()
STOP_TOKEN = "<end_of_turn>"
ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")
NUM_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def norm(s: str) -> str:
    """Canonical numeric string. Model samples can produce absurd magnitudes,
    inf and nan; those must compare unequal to the gold, not crash the writer."""
    s = str(s).strip().replace(",", "").replace("$", "")
    try:
        f = float(s)
        if f != f or f in (float("inf"), float("-inf")) or abs(f) > 1e15:
            return "__nonfinite__" + s
        return str(int(f)) if f == int(f) else f"{f:.6g}"
    except (ValueError, OverflowError):
        return s


def fewshot_block(q: str, a: str) -> str:
    reasoning, _, target = a.rpartition("####")
    return f"{q}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {target.strip()}"


def load_problems(args, rng):
    """(question, gold) pairs that are NOT the benchmark test set."""
    from datasets import load_dataset
    import pyarrow.parquet as pq

    if args.problems_file and Path(args.problems_file).exists():
        items = [tuple(x) for x in json.loads(Path(args.problems_file).read_text())]
        rng.shuffle(items)
        return items

    items = []
    if args.use_gsm_train:
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        for r in gsm:
            items.append((r["question"].strip(), norm(r["answer"].rsplit("####", 1)[1])))
    omi = Path("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
               "469216e3f46f4dacf476b382e192485ea51a143e/data")
    seen = set()
    for sh in args.shards:
        f = omi / f"train-{sh:05d}-of-00032.parquet"
        if not f.exists():
            print("missing", f)
            continue
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=20000,
                                     columns=["problem", "expected_answer", "problem_source"]):
            for r in batch.to_pylist():
                if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                    continue
                a = r["expected_answer"]
                if not a or not NUM_RE.match(a.strip()):
                    continue
                q = r["problem"].strip()
                h = hashlib.md5(q.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                items.append((q, norm(a)))
    rng.shuffle(items)
    if args.problems_file:
        Path(args.problems_file).write_text(json.dumps(items))
        print("cached problems ->", args.problems_file)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--n-problems", type=int, default=40000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--shards", type=int, nargs="*", default=[3])
    ap.add_argument("--use-gsm-train", action="store_true")
    ap.add_argument("--problems-file", default=None,
                    help="cache the (question, gold) pool here; reused if it exists")
    ap.add_argument("--build-problems-only", action="store_true")
    ap.add_argument("--fewshot-rows", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    items = load_problems(args, rng)
    if args.build_problems_only:
        print("built problem pool:", len(items))
        return
    items = items[: args.n_problems]
    print("problems:", len(items), flush=True)

    prompts = []
    for q, _ in items:
        user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": user}], chat_template=TEMPLATE,
            tokenize=False, add_generation_prompt=True))

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              max_model_len=1536, enforce_eager=False, seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    kept, n_gen, n_ok, solved = 0, 0, 0, 0
    fh = open(args.out, "w")
    pool = None
    if args.fewshot_rows:
        from datasets import load_dataset
        g = load_dataset("openai/gsm8k", "main", split="train")
        pool = [(r["question"], r["answer"]) for r in g]
    fewshot_left = args.fewshot_rows
    per_problem_stats = []
    for (q, gold), o in zip(items, outs):
        cands, seen_txt = [], set()
        n_right = 0
        for c in o.outputs:
            n_gen += 1
            txt = c.text.strip()
            m = ANS_RE.search(txt)
            if not m:
                continue
            if norm(m.group(1)) != gold:
                continue
            n_right += 1
            # exactly one marker, nothing numeric after it
            if txt.count("ANSWER:") != 1:
                continue
            body = txt[: m.start()].rstrip()
            if len(body) < 20:
                continue
            clean = f"{body}\n\nANSWER: {gold}"
            key = hashlib.md5(re.sub(r"\s+", " ", clean).encode()).hexdigest()
            if key in seen_txt:
                continue
            seen_txt.add(key)
            cands.append(clean)
        per_problem_stats.append(n_right / max(len(o.outputs), 1))
        if n_right:
            solved += 1
        n_ok += n_right
        cands.sort(key=len)                      # prefer the shorter derivations
        for clean in cands[: args.keep_per_problem]:
            prefix = ""
            nshot = 0
            if fewshot_left > 0 and pool:
                nshot = 10
                prefix = "\n\n".join(fewshot_block(a, b) for a, b in rng.sample(pool, 10)) + "\n\n"
                fewshot_left -= 1
            user = prefix + MATH_PROMPT_TEMPLATE.format(prompt=q)
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": user}], chat_template=TEMPLATE,
                tokenize=False, add_generation_prompt=True)
            fh.write(json.dumps({"prompt": prompt, "completion": clean + STOP_TOKEN,
                                 "source": "rft", "answer": gold, "nshot": nshot,
                                 "pass_rate": n_right / max(len(o.outputs), 1),
                                 "qid": hashlib.md5(q.encode()).hexdigest()}) + "\n")
            kept += 1
    fh.close()
    stats = {
        "problems": len(items), "generations": n_gen, "correct_generations": n_ok,
        "pass_rate": n_ok / max(n_gen, 1),
        "problems_solved_at_least_once": solved,
        "solve_rate": solved / max(len(items), 1),
        "rows_written": kept,
    }
    print(json.dumps(stats, indent=2))
    if args.stats_out:
        Path(args.stats_out).write_text(json.dumps(stats, indent=2))
    # per-problem pass rate, for picking hard problems later
    Path(args.out + ".passrate.json").write_text(json.dumps(
        [{"q": q, "gold": g, "pass": p} for (q, g), p in zip(items, per_problem_stats)]))


if __name__ == "__main__":
    main()
