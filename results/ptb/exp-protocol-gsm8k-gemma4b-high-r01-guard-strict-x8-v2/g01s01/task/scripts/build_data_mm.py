"""Second-stage corpus: MetaMathQA GSM subsets + extra OpenMathInstruct-2 solutions.

MetaMathQA's GSM_* rows are augmentations (answer-augmentation, rephrasing,
self-verification, FOBAR) of the GSM8K *train* split only. Output is the same
{prompt, completion, text, src} jsonl shape as build_data.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402
from build_data import OMI_DIR, NUM_RE, norm_q, fewshot_pool  # noqa: E402

TAIL_RE = re.compile(r"\n?The answer is:?\s*(.+?)\s*$", re.S)


def load_metamath(want, seed=0):
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    by = defaultdict(list)
    for r in ds:
        t = r["type"]
        if t not in want:
            continue
        by[t].append(r)
    rng = random.Random(seed)
    out = []
    dropped = 0
    for t, n in want.items():
        rows = by[t]
        rng.shuffle(rows)
        taken = 0
        for r in rows:
            if taken >= n:
                break
            resp = r["response"]
            m = TAIL_RE.search(resp)
            if not m:
                dropped += 1
                continue
            ans = m.group(1).strip().rstrip(".").replace(",", "").replace("$", "")
            if not NUM_RE.match(ans):
                dropped += 1
                continue
            body = resp[: m.start()]
            tgt = fmt.make_target(body, ans)
            if tgt is None or len(tgt) < 30:
                dropped += 1
                continue
            out.append({"question": r["query"].strip(), "target": tgt, "src": t})
            taken += 1
        print(f"  {t}: {taken}", flush=True)
    print("metamath dropped (no numeric 'The answer is'):", dropped, flush=True)
    return out


def load_omi_extra(shards, n_want, skip_per_problem, cap_per_problem):
    """Solutions beyond the first `skip_per_problem` already used for stage 1."""
    seen = Counter()
    out = []
    for sh in shards:
        path = os.path.join(OMI_DIR, f"train-{sh:05d}-of-00032.parquet")
        if not os.path.exists(path):
            continue
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=20000,
                                     columns=["problem", "generated_solution", "expected_answer", "problem_source"]):
            for r in batch.to_pylist():
                if r["problem_source"] != "augmented_gsm8k":
                    continue
                ans = (r["expected_answer"] or "").strip()
                if not NUM_RE.match(ans):
                    continue
                k = norm_q(r["problem"])
                seen[k] += 1
                if seen[k] <= skip_per_problem or seen[k] > cap_per_problem:
                    continue
                tgt = fmt.make_target(r["generated_solution"], ans)
                if tgt is None:
                    continue
                out.append({"question": r["problem"].strip(), "target": tgt, "src": "omi_extra"})
                if len(out) >= n_want:
                    return out
        print("  omi_extra after shard", sh, len(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ansaug", type=int, default=55000)
    ap.add_argument("--n-rephrased", type=int, default=55000)
    ap.add_argument("--n-sv", type=int, default=15000)
    ap.add_argument("--n-fobar", type=int, default=15000)
    ap.add_argument("--n-omi-extra", type=int, default=20000)
    ap.add_argument("--fewshot-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    want = {}
    for key, n in (("GSM_AnsAug", args.n_ansaug), ("GSM_Rephrased", args.n_rephrased),
                   ("GSM_SV", args.n_sv), ("GSM_FOBAR", args.n_fobar)):
        if n > 0:
            want[key] = n
    rows = load_metamath(want, seed=args.seed)
    if args.n_omi_extra > 0:
        rows += load_omi_extra(list(range(0, 10)), args.n_omi_extra,
                               skip_per_problem=2, cap_per_problem=5)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    pool = fewshot_pool()
    n_fs = 0
    with open(args.out, "w") as f:
        for r in rows:
            system = None
            if rng.random() < args.fewshot_frac:
                system = "\n\n".join(rng.sample(pool, rng.randint(1, 8)))
                n_fs += 1
            f.write(json.dumps({
                "prompt": fmt.render_prompt(r["question"], system),
                "completion": fmt.render_completion(r["target"]),
                "text": r["question"] + "\n\n" + r["target"],
                "src": r["src"],
            }) + "\n")
    print("wrote", len(rows), "rows to", args.out)
    print(Counter(r["src"] for r in rows))
    print("fewshot-prefixed rows:", n_fs)


if __name__ == "__main__":
    main()
