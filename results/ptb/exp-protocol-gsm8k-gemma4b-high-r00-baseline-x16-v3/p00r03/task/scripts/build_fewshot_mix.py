"""Build the few-shot-prefixed training mix for exp-03.

The grader always prepends a fixed 10-shot system message (2044 tokens) written
in GSM8K's own terse style, complete with <<48/2=24>> calculator annotations.
exp-02 was trained on bare prompts and loses 30.2 points under that prefix: it
imitates the demonstrations and keeps generating fresh problems after its own
answer.  These rows put a demonstration block in front of the question -- with a
varying number of shots, drawn from GSM8K train items outside dev500 -- so the
model learns to answer once and stop with demonstrations in context.

The prefix is baked into the `prompt` field exactly as
scripts/train_sft.py:render_prompt() would place a system message, so rendering
stays deterministic and cacheable.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TASK_DIR, read_jsonl, write_jsonl  # noqa: E402

SHOT_WEIGHTS = [(4, 0.15), (6, 0.15), (8, 0.20), (10, 0.50)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(TASK_DIR, "data", "sft_fs.jsonl"))
    ap.add_argument("--n-prefixed", type=int, default=20000)
    ap.add_argument("--n-plain", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--src", default=None, help="jsonl of prompt/completion rows; default is sft_v2 minus sft_v1")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = json.load(open(os.path.join(TASK_DIR, "data", "fewshot_pool.json")))

    if args.src:
        fresh = read_jsonl(args.src)
        rng.shuffle(fresh)
        print(f"rows from {args.src}: {len(fresh)}")
    else:
        v1 = read_jsonl(os.path.join(TASK_DIR, "data", "sft_v1.jsonl"))
        v2 = read_jsonl(os.path.join(TASK_DIR, "data", "sft_v2.jsonl"))
        seen = {(r["prompt"], r["completion"]) for r in v1}
        fresh = [r for r in v2 if (r["prompt"], r["completion"]) not in seen]
        rng.shuffle(fresh)
        print(f"rows never seen in exp-02: {len(fresh)}")

    need = args.n_prefixed + args.n_plain
    rows = fresh[:need]
    if len(rows) < need:
        print(f"only {len(rows)} rows available for the requested {need}")
        n_pref = int(round(len(rows) * args.n_prefixed / need))
        args.n_prefixed, args.n_plain = n_pref, len(rows) - n_pref
        print(f"rescaled to {args.n_prefixed} prefixed / {args.n_plain} plain")

    shots_choices = [s for s, _ in SHOT_WEIGHTS]
    shots_w = [w for _, w in SHOT_WEIGHTS]

    out = []
    for i, r in enumerate(rows):
        if i < args.n_prefixed:
            k = rng.choices(shots_choices, weights=shots_w)[0]
            demos = rng.sample(pool, k + 1)
            demos = [d for d in demos if r["question"] not in d][:k]
            system = "\n\n".join(demos)
            prompt = system.strip() + "\n\n" + r["prompt"]
            kind = f"fs{k}"
        else:
            prompt = r["prompt"]
            kind = "fs0"
        out.append(
            {
                "prompt": prompt,
                "completion": r["completion"],
                "question": r["question"],
                "answer": r["answer"],
                "source": r["source"] + "|" + kind,
            }
        )

    rng.shuffle(out)
    write_jsonl(args.out, out)
    from collections import Counter

    print(Counter(r["source"].split("|")[1] for r in out))
    print(f"wrote {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
