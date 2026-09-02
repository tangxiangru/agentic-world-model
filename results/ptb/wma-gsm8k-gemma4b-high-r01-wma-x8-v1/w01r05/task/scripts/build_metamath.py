#!/usr/bin/env python3
"""MetaMathQA's GSM slices, reformatted onto the single answer marker the grader reads.

Only GSM_AnsAug and GSM_Rephrased are taken. GSM_SV and GSM_FOBAR are deliberately
excluded: they ask 'what is the value of unknown variable x' or hand the answer to
the model in the prompt, neither of which is the GSM8K task shape.

MetaMathQA responses end with BOTH '#### N' and 'The answer is: N'. That is the
double_answer_format pitfall verbatim, so both are stripped and one
'ANSWER: N<end_of_turn>' is appended.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import (  # noqa: E402
    MATH_PROMPT_TEMPLATE,
    STOP_TOKEN,
    gsm8k_train_shots,
    load_shot_pool,
    norm_num,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = (
    "/home/ben/hf_cache/hub/datasets--meta-math--MetaMathQA/snapshots/"
    "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18/MetaMathQA-395K.json"
)
KEEP_TYPES = ("GSM_AnsAug", "GSM_Rephrased")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120000)
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    ap.add_argument("--max-shots", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "data/mm_gsm.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    load_shot_pool()
    data = json.load(open(SRC))
    rows, stats = [], {"seen": 0, "drop_type": 0, "drop_marker": 0, "drop_num": 0, "dupe": 0}
    seen = set()
    for r in data:
        stats["seen"] += 1
        if r["type"] not in KEEP_TYPES:
            stats["drop_type"] += 1
            continue
        resp = r["response"]
        m = re.search(r"\n?####\s*([^\n]*)", resp)
        if m is None:
            stats["drop_marker"] += 1
            continue
        gold = norm_num(m.group(1))
        if gold is None:
            stats["drop_num"] += 1
            continue
        body = resp[: m.start()].strip()
        # nothing may follow: 'The answer is: N' lives after the #### marker
        if "####" in body or "The answer is" in body:
            stats["drop_marker"] += 1
            continue
        key = (r["query"].strip(), body)
        if key in seen:
            stats["dupe"] += 1
            continue
        seen.add(key)
        rows.append({"problem": r["query"].strip(), "body": body, "answer": gold, "source": r["type"]})
    rng.shuffle(rows)
    rows = rows[: args.n]
    print("stats:", stats, "kept:", len(rows))

    n_fs = 0
    with open(args.out, "w") as f:
        for r in rows:
            user = MATH_PROMPT_TEMPLATE.replace("{prompt}", r["problem"])
            k = 0
            if rng.random() < args.fewshot_frac:
                k = rng.randint(1, args.max_shots)
                user = gsm8k_train_shots(rng, k) + "\n\n" + user
                n_fs += 1
            f.write(
                json.dumps(
                    {
                        "prompt": user,
                        "completion": f"{r['body']}\n\nANSWER: {r['answer']}{STOP_TOKEN}",
                        "answer": r["answer"],
                        "source": r["source"],
                        "n_shots": k,
                    }
                )
                + "\n"
            )
    print(f"wrote {len(rows)} rows to {args.out} ({n_fs} few-shot)")


if __name__ == "__main__":
    main()
