#!/usr/bin/env python3
"""Second-stage data: more OpenMathInstruct-2 augmented_gsm8k rows, disjoint from an existing file.

Same rendering and target shape as build_data.py (imported from it), so training and
grading still see the byte-identical string templates/gemma3.jinja produces.
"""
from __future__ import annotations

import argparse
import json
import random
import re

from datasets import load_dataset

from build_data import BOXED_RE, EOT, fewshot_text, load_gsm8k_train, render_prompt, sid, strip_boxed_tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=250000)
    ap.add_argument("--sources", nargs="*", default=["augmented_gsm8k", "gsm8k"])
    ap.add_argument("--fewshot-frac", type=float, default=0.04)
    ap.add_argument("--fewshot-max-k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    seen = set()
    for p in args.exclude:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                seen.add(sid(r["prompt"] + r["target"]))
    print(f"excluding {len(seen)} rows already used")

    keep = set(args.sources)
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train")
    ds = ds.filter(
        lambda b: [s in keep for s in b["problem_source"]], batched=True, num_proc=12
    ).shuffle(seed=args.seed)
    print(f"candidate pool: {len(ds)}")

    rng = random.Random(args.seed)
    fs_pool = load_gsm8k_train()
    out, dedup = [], set()
    with open(args.out, "w") as f:
        n = 0
        for r in ds:
            ans = str(r["expected_answer"]).strip().replace(",", "")
            if not re.fullmatch(r"-?\d+(\.\d+)?", ans):
                continue
            body = strip_boxed_tail(r["generated_solution"])
            if body is None or BOXED_RE.search(body):
                continue
            k = sid(r["problem"].strip() + "|" + body + "|" + ans)
            if k in dedup:
                continue
            dedup.add(k)
            block = None
            if rng.random() < args.fewshot_frac:
                shots = rng.sample(fs_pool, rng.randint(1, args.fewshot_max_k))
                block = "\n\n".join(
                    fewshot_text(s["question"], s["reasoning"], s["gold"]) for s in shots
                )
            prompt = render_prompt(r["problem"].strip(), block)
            target = f"{body.strip()}\n\nANSWER: {ans}{EOT}"
            if sid(prompt + target) in seen:
                continue
            f.write(
                json.dumps(
                    {
                        "id": sid(prompt + target),
                        "source": r["problem_source"],
                        "prompt": prompt,
                        "target": target,
                        "gold": ans,
                    }
                )
                + "\n"
            )
            n += 1
            if n >= args.n:
                break
    print(f"wrote {args.out} rows={n}")


if __name__ == "__main__":
    main()
