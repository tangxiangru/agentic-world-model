#!/usr/bin/env python3
"""Turn kept RFT samples (scripts/gen_rft.py output) into prompt/completion rows
with the same few-shot prefix policy as scripts/build_data.py, optionally mixed
with a slice of the original SFT file.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_data import (  # noqa: E402
    CALC_RE,
    EOT,
    fewshot_block,
    render_completion,
    render_prompt,
)

ANSWER_TAIL = re.compile(r"\n*\s*ANSWER:\s*\$?-?[\d,]+(?:\.\d+)?\s*$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix-sft", default=None, help="jsonl of already-rendered rows")
    ap.add_argument("--mix-sft-n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--p-fewshot-short", type=float, default=0.30)
    ap.add_argument("--p-fewshot-full", type=float, default=0.08)
    ap.add_argument("--eval-fewshot-file", default="data/fewshot_system.txt")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    eval_fewshot = Path(args.eval_fewshot_file).read_text()

    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    shot_pool = [
        {
            "question": r["question"].strip(),
            "reasoning_raw": "####".join(r["answer"].split("####")[:-1]).strip(),
            "answer": r["answer"].split("####")[-1].strip(),
        }
        for r in gsm
    ]

    rows = []
    n_bad = 0
    with open(args.rft) as f:
        for line in f:
            r = json.loads(line)
            sol = r["solution"].strip()
            # the sampled text already ends in "ANSWER: N"; strip it so
            # render_completion re-attaches exactly one marker
            body = ANSWER_TAIL.sub("", sol).strip()
            if not body or "ANSWER:" in body:
                n_bad += 1
                continue
            body = CALC_RE.sub("", body)
            rows.append({"question": r["question"], "reasoning": body, "answer": r["answer"]})
    print(f"[rft] {len(rows)} usable rows, {n_bad} dropped (empty body or extra ANSWER marker)")

    out_rows = []
    for rec in rows:
        u = rng.random()
        if u < args.p_fewshot_full:
            prefix = eval_fewshot
        elif u < args.p_fewshot_full + args.p_fewshot_short:
            shots = rng.sample(shot_pool, rng.randint(1, 4))
            prefix = "\n\n".join(
                fewshot_block(s["question"], s["reasoning_raw"], s["answer"]) for s in shots
            )
        else:
            prefix = None
        out_rows.append(
            {
                "prompt": render_prompt(rec["question"], prefix),
                "completion": render_completion(rec["reasoning"], rec["answer"]),
                "src": "rft",
            }
        )

    if args.mix_sft and args.mix_sft_n:
        sft = [json.loads(l) for l in open(args.mix_sft)]
        rng.shuffle(sft)
        out_rows.extend(sft[: args.mix_sft_n])
        print(f"[mix] added {min(args.mix_sft_n, len(sft))} rows from {args.mix_sft}")

    rng.shuffle(out_rows)
    bad_tail = sum(1 for r in out_rows if not r["completion"].endswith(EOT))
    assert bad_tail == 0, bad_tail
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
