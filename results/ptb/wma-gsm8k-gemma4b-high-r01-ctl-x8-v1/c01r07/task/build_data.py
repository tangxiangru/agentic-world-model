#!/usr/bin/env python3
"""Build an SFT jsonl whose prompts are byte-identical to what the grader renders.

Sources (all derived from the GSM8K *train* split only; the test split is never read):
  * nvidia/OpenMathInstruct-2, train_1M shard, problem_source in {gsm8k, augmented_gsm8k}
  * openai/gsm8k train split (gold, terse, calculator annotations stripped)

Output rows: {"prompt": <rendered up to '<start_of_turn>model\\n'>,
              "completion": <body + '\\n\\nANSWER: N' + '<end_of_turn>\\n'>,
              "source": ..., "fewshot": bool}
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
from collections import defaultdict

import pyarrow.dataset as ds

import prompt_spec as ps

OMI_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"
GSM_GLOB = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"

NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUM_RE.match(a):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    return a


def qkey(q: str) -> str:
    return hashlib.md5(re.sub(r"\W+", " ", q.lower()).strip().encode()).hexdigest()


def load_omi(max_per_problem: int):
    files = sorted(glob.glob(OMI_GLOB))
    assert files, OMI_GLOB
    tbl = ds.dataset(files).to_table(
        filter=ds.field("problem_source").isin(["augmented_gsm8k", "gsm8k"])
    )
    by_problem = defaultdict(list)
    for r in tbl.to_pylist():
        ans = norm_answer(r["expected_answer"])
        if ans is None:
            continue
        body = ps.clean_body(r["generated_solution"])
        if not body or len(body) < 40:
            continue
        by_problem[r["problem"]].append((body, ans, r["problem_source"]))
    out = []
    rng = random.Random(0)
    for prob, sols in by_problem.items():
        rng.shuffle(sols)
        seen = set()
        kept = 0
        for body, ans, src in sols:
            h = hashlib.md5(body.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            out.append({"question": prob, "body": body, "answer": ans, "source": "omi2:" + src})
            kept += 1
            if kept >= max_per_problem:
                break
    return out


def load_gsm8k_train():
    files = sorted(glob.glob(GSM_GLOB))
    assert files, GSM_GLOB
    tbl = ds.dataset(files).to_table()
    out = []
    for r in tbl.to_pylist():
        a = r["answer"]
        ans = norm_answer(a.split("####")[-1])
        if ans is None:
            continue
        body = ps.clean_body(a)
        if not body:
            continue
        out.append({"question": r["question"], "body": body, "answer": ans,
                    "source": "gsm8k:train_gold"})
    return out


def load_heldout() -> set:
    """Questions reserved for the local probe set; never trained on."""
    with open("data/heldout_questions.json") as f:
        return {qkey(q) for q in json.load(f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    held = load_heldout()
    omi = [r for r in load_omi(args.max_per_problem) if qkey(r["question"]) not in held]
    gold = [r for r in load_gsm8k_train() if qkey(r["question"]) not in held]
    print(f"omi rows {len(omi)}  gsm8k-gold rows {len(gold)}  (held out {len(held)} probe questions)")

    rng.shuffle(omi)
    budget = max(0, args.n - len(gold))
    rows = gold + omi[:budget]
    rng.shuffle(rows)

    sysmsg = ps.fewshot_system_message()
    print("template sha256:", ps.template_sha256())

    n_fs = int(len(rows) * args.fewshot_frac)
    dropped = 0
    written = 0
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            fewshot = i < n_fs
            prompt = ps.render_prompt(r["question"], sysmsg if fewshot else None)
            completion = ps.render_target(r["body"], r["answer"])
            # one answer marker, no competing marker the grader might read first
            if (completion.count(ps.ANSWER_MARKER) != 1
                    or "####" in completion or "boxed" in completion):
                dropped += 1
                continue
            assert completion.endswith(ps.END + "\n")
            f.write(json.dumps({"prompt": prompt, "completion": completion,
                                "source": r["source"], "fewshot": fewshot,
                                "question": r["question"], "answer": r["answer"]}) + "\n")
            written += 1
    print(f"wrote {written} rows to {args.out} "
          f"({n_fs} slots with the 10-shot prefix, {dropped} dropped on marker check)")


if __name__ == "__main__":
    main()
