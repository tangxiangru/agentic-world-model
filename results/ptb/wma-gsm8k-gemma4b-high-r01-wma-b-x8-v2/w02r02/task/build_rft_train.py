#!/usr/bin/env python3
"""Assemble the round-2 training file from the rejection-sampling output.

Three components, all from GSM8K *train*-derived problems:
  A. on-policy: the model's own solutions that reached the gold answer
  B. hard tail: for problems where 0 of k samples were right, the reference
     OpenMathInstruct-2 solution -- the model cannot learn those from itself
  C. a long few-shot slice (k = 4..10 demos) so the ~1700-token 10-shot context
     the grader always sends is in-distribution; exp-02 trained at k = 2..4 only

Reproduces gen_rft.py's problem subset exactly (same file, same Random(seed)).
"""
import argparse
import json
import random
from pathlib import Path

STOP_TOKEN = "<end_of_turn>"
_HEAD = 'Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
_TAIL = '\n\nRemember to put your answer on its own line at the end'


def raw_question(prompt: str) -> str:
    """Recover the bare problem text from the inspect_evals prompt template."""
    assert prompt.startswith(_HEAD) and _TAIL in prompt, prompt[:120]
    return prompt[len(_HEAD):prompt.index(_TAIL)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", default="data/rft.jsonl")
    ap.add_argument("--problems", default="data/sft_train2.jsonl")
    ap.add_argument("--n-problems", type=int, default=24000)
    ap.add_argument("--gen-seed", type=int, default=0, help="the seed gen_rft.py used")
    ap.add_argument("--hard-tail-max", type=int, default=8000)
    ap.add_argument("--fewshot-frac", type=float, default=0.20)
    ap.add_argument("--kmin", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default="data/rft_train.jsonl")
    args = ap.parse_args()

    # A. on-policy correct samples
    rft = [json.loads(l) for l in open(args.rft)]
    solved_prompts = {r["prompt"] for r in rft}

    # B. the problems the sampler never solved, with their reference solution
    allrows = [json.loads(l) for l in open(args.problems)]
    zs = [r for r in allrows if not r.get("system")]
    random.Random(args.gen_seed).shuffle(zs)
    sampled = zs[: args.n_problems]
    unsolved = [r for r in sampled if r["prompt"] not in solved_prompts]

    rng = random.Random(args.seed)
    rng.shuffle(unsolved)
    hard = [
        {"system": None, "prompt": r["prompt"], "completion": r["completion"], "answer": r["answer"]}
        for r in unsolved[: args.hard_tail_max]
    ]

    rows = rft + hard
    rng.shuffle(rows)

    # C. long few-shot prefixes, demos drawn from this same pool
    n_fs = int(len(rows) * args.fewshot_frac)
    pool = rows[n_fs:]
    out = []
    for i, r in enumerate(rows):
        system = None
        if i < n_fs and len(pool) > args.kmax:
            k = rng.randint(args.kmin, args.kmax)
            demos = rng.sample(pool, k)
            parts = []
            for d in demos:
                body = d["completion"].replace(STOP_TOKEN, "").rsplit("\n\nANSWER:", 1)[0].rstrip()
                q = raw_question(d["prompt"])
                parts.append(f"{q}\n\nReasoning:\n{body}\n\nANSWER: {d['answer']}")
            system = "\n\n".join(parts)
        out.append({"system": system, "prompt": r["prompt"],
                    "completion": r["completion"], "answer": r["answer"]})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({
        "on_policy_rows": len(rft),
        "hard_tail_rows": len(hard),
        "unsolved_problems": len(unsolved),
        "total_rows": len(out),
        "fewshot_rows": n_fs,
    }, indent=2))


if __name__ == "__main__":
    main()
