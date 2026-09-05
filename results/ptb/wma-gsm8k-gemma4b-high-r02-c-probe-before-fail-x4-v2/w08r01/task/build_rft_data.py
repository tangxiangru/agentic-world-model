"""Rejection-sampling fine-tuning data: sample the current checkpoint on GSM8K
TRAIN problems, keep only the samples whose final answer is right, and write them
in the same target format as the SFT corpus.

Problems come from GSM8K train rows 0..7172 and from OpenMathInstruct-2's
augmented gsm8k problems (both train-derived). Rows 7173+ stay held out.
No test-split item is involved at any point.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re

import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from build_sft_data import HOLDOUT_START, norm_answer
from eval_format import fewshot_system_message, load_template, user_prompt
from probe_eval import last_number


def collect_problems(n_aug: int, seed: int) -> list[dict]:
    out = []
    d = load_dataset("openai/gsm8k", "main")["train"]
    for i in range(HOLDOUT_START):
        r = d[i]
        ans = norm_answer(r["answer"].split("####")[-1])
        if ans is not None:
            out.append({"question": r["question"], "gold": ans, "src": "gsm8k_train"})
    if n_aug:
        files = sorted(glob.glob(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"))
        seen = set()
        aug = []
        for f in files:
            t = pq.read_table(f, columns=["problem", "expected_answer", "problem_source"])
            for r in t.to_pylist():
                if r["problem_source"] != "augmented_gsm8k":
                    continue
                q = r["problem"].strip()
                if q in seen:
                    continue
                ans = norm_answer(r["expected_answer"])
                if ans is None:
                    continue
                seen.add(q)
                aug.append({"question": q, "gold": ans, "src": "augmented_gsm8k"})
        random.Random(seed).shuffle(aug)
        out += aug[:n_aug]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-aug", type=int, default=30000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--sample-fewshot", type=int, default=1,
                    help="1 = sample under the grader's 10-shot prefix (vLLM prefix caching makes the "
                         "shared 2044-token prefix nearly free); 0 = zero-shot")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    probs = collect_problems(args.n_aug, args.seed)
    print(f"{len(probs)} problems to sample")

    tok = AutoTokenizer.from_pretrained(args.model)
    tpl = load_template()
    sysmsg = fewshot_system_message()
    def sample_msgs(q):
        m = [{"role": "system", "content": sysmsg}] if args.sample_fewshot else []
        return m + [{"role": "user", "content": user_prompt(q)}]

    prompts = [tok.apply_chat_template(sample_msgs(p["question"]), chat_template=tpl,
                                       tokenize=False, add_generation_prompt=True) for p in probs]

    max_len = 3072 if args.sample_fewshot else 1024
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=max_len,
              enable_prefix_caching=True, disable_log_stats=True)
    # no per-request seed: a per-request seed forces vLLM to keep a private RNG per sequence
    # and measurably cuts batched-sampling throughput. Reproducibility here comes from the
    # kept data file and the recorded stats, not from the sampler.
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens)
    outs = llm.generate(prompts, sp)

    n_kept = 0
    n_solved = 0
    n_trunc = 0
    n_wrong = 0
    rows = []
    for p, o in zip(probs, outs):
        good = []
        for c in o.outputs:
            t = c.text.strip()
            if c.finish_reason != "stop":
                n_trunc += 1
                continue
            if t.count("ANSWER:") != 1 or last_number(t) != p["gold"]:
                n_wrong += 1
                continue
            good.append(t)
        if good:
            n_solved += 1
        # keep the most typical correct solutions for this problem (closest to the median
        # length of its own correct samples): shortest-first would bias towards skipped
        # steps, longest-first towards the rambling that causes greedy repetition loops
        uniq = sorted(set(good))
        if uniq:
            med = sorted(len(x) for x in uniq)[len(uniq) // 2]
            uniq = sorted(uniq, key=lambda x: abs(len(x) - med))
        good = uniq[: args.keep_per_problem]
        for g in good:
            rows.append({"question": p["question"], "target": g, "src": "rft:" + p["src"],
                         "gold": p["gold"]})
            n_kept += 1

    rng.shuffle(rows)
    n_fs = 0
    with open(args.out, "w") as f:
        for r in rows:
            use_fs = rng.random() < args.fewshot_frac
            n_fs += use_fs
            msgs = []
            if use_fs:
                msgs.append({"role": "system", "content": sysmsg})
            msgs.append({"role": "user", "content": user_prompt(r["question"])})
            msgs.append({"role": "assistant", "content": r["target"]})
            f.write(json.dumps({"messages": msgs, "src": r["src"], "fewshot": use_fs,
                                "target": r["target"].strip() + "<end_of_turn>",
                                "text": r["question"] + "\n" + r["target"]}) + "\n")
    stats = {"problems": len(probs), "solved_at_least_once": n_solved,
             "solve_rate": n_solved / len(probs), "rows_written": n_kept,
             "with_fewshot_prefix": n_fs, "k": args.k, "temperature": args.temperature,
             "sampled_under_fewshot": bool(args.sample_fewshot),
             "n_truncated": n_trunc, "n_wrong_answer": n_wrong}
    print(json.dumps(stats, indent=2))
    if args.stats:
        json.dump(stats, open(args.stats, "w"), indent=2)


if __name__ == "__main__":
    main()
