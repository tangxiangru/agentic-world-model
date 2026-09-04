"""Problem pool for rejection sampling: {id, question, gold}.

  * every gsm8k TRAIN problem in [0:7273] (the eval's own distribution;
    7273-7472 stay held out as the probe set)
  * OpenMathInstruct-2 gsm8k / augmented_gsm8k problems that are NOT already in
    the SFT file, so the sampler is asked things the SFT run did not memorise

Only the reference ANSWER is taken from these sources - the chains come from the
checkpoint being improved.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

HELD_OUT_FROM = 7273
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def norm_num(s):
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    if not NUM_RE.match(s):
        return None
    f = float(s)
    return str(int(f)) if f == int(f) else str(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sft-file", default="/home/ben/task/work/data/sft_v2.jsonl")
    ap.add_argument("--extra-omi2", type=int, default=15000)
    ap.add_argument("--extra-metamath", type=int, default=18000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    used = set()
    for line in open(args.sft_file):
        used.add(" ".join(json.loads(line)["question"].split()).lower())

    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")["train"]
    rows = []
    for i in range(HELD_OUT_FROM):
        r = ds[i]
        g = norm_num(r["answer"].split("####")[-1])
        if g is not None:
            rows.append({"id": f"gsm8k-{i}", "question": r["question"], "gold": g})
    n_gsm = len(rows)

    import pyarrow.parquet as pq

    fresh, seen = [], set()
    for f in sorted(glob.glob("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
                              "snapshots/*/data/train_1M-*.parquet")):
        for r in pq.read_table(f, columns=["problem", "expected_answer",
                                           "problem_source"]).to_pylist():
            if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                continue
            g = norm_num(r["expected_answer"])
            if g is None:
                continue
            k = " ".join(r["problem"].split()).lower()
            if k in used or k in seen:
                continue
            seen.add(k)
            fresh.append({"id": f"omi2fresh-{len(fresh)}", "question": r["problem"].strip(),
                          "gold": g})
    random.Random(args.seed).shuffle(fresh)
    fresh = fresh[: args.extra_omi2]
    rows += fresh

    # MetaMathQA's GSM_AnsAug / GSM_Rephrased are augmentations of GSM8K TRAIN
    # problems (never test). Only the reference answer is taken, from the
    # "The answer is: X" tail; the chain will come from our own checkpoint.
    mm = []
    mm_files = glob.glob("/home/ben/hf_cache/hub/datasets--meta-math--MetaMathQA/"
                         "snapshots/*/MetaMathQA-395K.json")
    if mm_files and args.extra_metamath:
        for r in json.load(open(mm_files[0])):
            if r["type"] not in ("GSM_AnsAug", "GSM_Rephrased"):
                continue
            m = re.search(r"The answer is:\s*(.+?)\s*$", r["response"])
            if not m:
                continue
            g = norm_num(m.group(1))
            if g is None:
                continue
            q = r["query"].strip()
            k = " ".join(q.split()).lower()
            if k in used or k in seen:
                continue
            seen.add(k)
            mm.append({"id": f"metamath-{len(mm)}", "question": q, "gold": g})
        random.Random(args.seed + 1).shuffle(mm)
        mm = mm[: args.extra_metamath]
        rows += mm

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} problems ({n_gsm} gsm8k-train + {len(fresh)} unseen omi2 "
          f"+ {len(mm)} metamath-gsm) to {args.out}")


if __name__ == "__main__":
    main()
