"""Render {problem, body, answer} rows into grader-formatted {prompt, completion}."""
from __future__ import annotations

import argparse
import json
import random
import sys

sys.path.insert(0, "/home/ben/task/scripts")
from common import (grader_fewshot_system, get_tokenizer,  # noqa: E402
                    render_completion, render_prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    ap.add_argument("--max-sol-tokens", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = get_tokenizer()
    sysmsg = grader_fewshot_system()

    rows = []
    for path in args.inp:
        for line in open(path):
            rows.append(json.loads(line))
    rng.shuffle(rows)
    n_few = int(len(rows) * args.fewshot_frac)

    n_long = 0
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            completion = render_completion(r["body"], str(r["answer"]))
            if len(tok(completion, add_special_tokens=False)["input_ids"]) > args.max_sol_tokens:
                n_long += 1
                continue
            use_few = i < n_few
            f.write(json.dumps({
                "prompt": render_prompt(tok, r["problem"], system=sysmsg if use_few else None),
                "completion": completion,
                "fewshot": use_few,
                "src": r.get("src", "?"),
                "target": completion,
            }) + "\n")
    print(f"wrote {args.out}; dropped {n_long} over-long", flush=True)


if __name__ == "__main__":
    main()
