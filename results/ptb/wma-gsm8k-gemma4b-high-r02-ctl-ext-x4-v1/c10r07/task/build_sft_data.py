#!/usr/bin/env python3
"""Build the SFT file for exp-02 from OpenMathInstruct-2's GSM8K-derived rows.

Every row is written in the grader's own rendering (common.render_prompt) with a
single answer marker at the very end followed by <end_of_turn>, so the two
pitfalls that cost a whole run -- eos_mismatch and double_answer_format --
cannot occur. Rows longer than --max-seq-len are dropped rather than truncated
(pitfall seq_len_truncation).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import common

OMI2 = "nvidia/OpenMathInstruct-2"
OMI2_REV = "469216e3"
NUMERIC = re.compile(r"^-?\d+(\.\d+)?$")


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{X} with X (brace-matched). Leaves nothing else behind."""
    out, i = [], 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        k, depth = j + len("\\boxed{"), 1
        while k < len(text) and depth:
            depth += (text[k] == "{") - (text[k] == "}")
            k += 1
        out.append(text[j + len("\\boxed{"): k - 1])
        i = k


def clean_solution(sol: str) -> str:
    sol = strip_boxed(sol)
    sol = sol.replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")
    sol = re.sub(r"\n{3,}", "\n\n", sol)
    return sol.strip()


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUMERIC.match(a):
        return None
    if "." in a:  # 18.0 -> 18
        a = a.rstrip("0").rstrip(".")
    return a or None


def load_rows(shards: list[int]) -> list[dict]:
    rows = []
    for s in shards:
        path = hf_hub_download(
            OMI2, f"data/train-{s:05d}-of-00032.parquet", repo_type="dataset"
        )
        tbl = pq.read_table(path, columns=["problem", "generated_solution",
                                           "expected_answer", "problem_source"])
        src = tbl.column("problem_source").to_pylist()
        prob = tbl.column("problem").to_pylist()
        sol = tbl.column("generated_solution").to_pylist()
        ans = tbl.column("expected_answer").to_pylist()
        for so, p, g, a in zip(src, prob, sol, ans):
            if so in ("gsm8k", "augmented_gsm8k"):
                rows.append({"source": so, "problem": p, "solution": g, "answer": a})
        print(f"  shard {s}: cumulative {len(rows)} gsm8k-derived rows", flush=True)
    return rows


def fewshot_prefix_k(rng: random.Random, pool: list[str], k: int) -> str:
    return "\n\n".join(rng.sample(pool, k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--raw-out", default="data/sft_v1_raw.jsonl")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--per-problem", type=int, default=2)
    ap.add_argument("--gsm8k-per-problem", type=int, default=3)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--max-completion-tokens", type=int, default=640)
    ap.add_argument("--p-full-fewshot", type=float, default=0.04)
    ap.add_argument("--p-short-fewshot", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-questions", default=None,
                    help="jsonl of {question}: held out of training so it stays a clean probe")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(common.BASE_SNAPSHOT)
    print("chat template sha:", common.chat_template_sha(), flush=True)

    rows = load_rows(args.shards)
    print("raw gsm8k-derived rows:", len(rows), Counter(r["source"] for r in rows))

    excl = set()
    if args.exclude_questions:
        excl = {json.loads(l)["question"].strip() for l in open(args.exclude_questions)}
        before = len(rows)
        rows = [r for r in rows if r["problem"].strip() not in excl]
        print(f"excluded {before - len(rows)} rows matching {len(excl)} held-out probe questions")

    # ---- clean + cap solutions per problem -------------------------------
    by_problem: dict[str, list[dict]] = defaultdict(list)
    dropped = Counter()
    for r in rows:
        a = norm_answer(r["answer"])
        if a is None:
            dropped["non_numeric_answer"] += 1
            continue
        body = clean_solution(r["solution"])
        if not body or "ANSWER:" in body or "####" in body:
            dropped["bad_body"] += 1
            continue
        if "\\frac" in body or "\\sqrt" in body or "\\text" in body:
            dropped["latex_heavy"] += 1
            continue
        by_problem[r["problem"]].append({"problem": r["problem"], "body": body,
                                         "answer": a, "source": r["source"]})
    print("unique problems:", len(by_problem), "dropped:", dict(dropped))

    # prefer short-but-not-trivial solutions; the cap is per source, because the
    # 7,473 original gsm8k problems are the benchmark's own distribution and are
    # worth more repetitions than the augmented variants.
    pool = []
    for prob, cands in by_problem.items():
        cands.sort(key=lambda c: len(c["body"]))
        mid = cands[len(cands) // 4:] or cands  # drop the shortest quartile (often degenerate)
        cap = args.gsm8k_per_problem if mid[0]["source"] == "gsm8k" else args.per_problem
        pool.extend(mid[:cap])
    rng.shuffle(pool)
    print("candidate pool:", len(pool), Counter(c["source"] for c in pool))

    # ---- few-shot pool for the robustness rows ---------------------------
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import DATASET_PATH, record_to_sample, sample_to_fewshot

    shot_pool = [
        sample_to_fewshot(s)
        for s in hf_dataset(path=DATASET_PATH, data_dir="main", split="train",
                            sample_fields=record_to_sample, shuffle=True,
                            seed=7, limit=400)
    ]
    full_fewshot = common.fewshot_system_message()

    # ---- render ----------------------------------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    kept, n_too_long, lens = 0, 0, []
    kind_counts = Counter()
    with open(args.out, "w") as f, open(args.raw_out, "w") as fr:
        for c in pool:
            if kept >= args.n:
                break
            completion = common.format_target(c["body"], c["answer"])
            ctoks = tok.encode(completion, add_special_tokens=False)
            if len(ctoks) > args.max_completion_tokens:
                n_too_long += 1
                continue

            u = rng.random()
            if u < args.p_full_fewshot:
                sysmsg, kind = full_fewshot, "fewshot10"
            elif u < args.p_full_fewshot + args.p_short_fewshot:
                sysmsg = fewshot_prefix_k(rng, shot_pool, rng.choice([1, 2, 3]))
                kind = "fewshot_short"
            else:
                sysmsg, kind = None, "zeroshot"

            messages = ([{"role": "system", "content": sysmsg}] if sysmsg else [])
            messages.append({"role": "user", "content": common.user_message(c["problem"])})
            prompt = tok.apply_chat_template(
                messages, chat_template=common.chat_template(),
                tokenize=False, add_generation_prompt=True,
            )
            ptoks = tok.encode(prompt, add_special_tokens=False)
            total = len(ptoks) + len(ctoks)
            if total > args.max_seq_len:
                n_too_long += 1
                continue

            f.write(json.dumps({"prompt": prompt, "completion": completion,
                                "kind": kind, "n_tokens": total}) + "\n")
            fr.write(json.dumps({"question": c["problem"],
                                 "answer": c["body"] + f"\n\nANSWER: {c['answer']}"}) + "\n")
            lens.append(total)
            kind_counts[kind] += 1
            kept += 1

    lens.sort()
    print(f"wrote {kept} rows to {args.out} (dropped {n_too_long} over-length)")
    print("kinds:", dict(kind_counts))
    print(f"token len p50={lens[len(lens)//2]} p95={lens[int(len(lens)*0.95)]} max={lens[-1]}")
    print(f"total tokens per epoch: {sum(lens)/1e6:.2f}M")


if __name__ == "__main__":
    main()
