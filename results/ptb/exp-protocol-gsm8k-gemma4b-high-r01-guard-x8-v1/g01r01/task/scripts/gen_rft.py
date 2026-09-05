"""Rejection sampling: draw k solutions per problem from a checkpoint, keep the
ones whose final 'ANSWER: <n>' line matches the reference answer.

Problems come from the GSM8K TRAIN split and from OpenMathInstruct-2's
gsm8k/augmented_gsm8k problems (also TRAIN-derived). No test item is read.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

sys.path.insert(0, "/home/ben/task/scripts")
from common import (STOP_TOKEN, grader_fewshot_system, get_tokenizer,  # noqa: E402
                    render_completion, render_prompt)

ANS_RE = re.compile(r"ANSWER:\s*(-?[\d,]*\.?\d+)\s*$")


def norm(x: str) -> str | None:
    x = x.replace(",", "").strip().rstrip(".")
    try:
        v = float(x)
    except ValueError:
        return None
    return f"{v:.5g}"


def extract(text: str) -> str | None:
    lines = [ln for ln in text.strip().split("\n") if ln.strip()]
    if not lines:
        return None
    m = ANS_RE.search(lines[-1].strip())
    return norm(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--problems", required=True, help="jsonl of {problem, answer}")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--gpu-frac", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fewshot", type=int, default=1)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    probs = [json.loads(l) for l in open(args.problems)]
    tok = get_tokenizer()
    sysmsg = grader_fewshot_system() if args.fewshot else None
    prompts = [render_prompt(tok, p["problem"], system=sysmsg) for p in probs]
    print(f"[gen] {len(prompts)} problems x k={args.k}", flush=True)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              max_model_len=3072, seed=args.seed, enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    n_correct = n_rows = 0
    solved = 0
    stats = []
    with open(args.out, "w") as f:
        for p, o in zip(probs, outs):
            gold = norm(str(p["answer"]))
            kept, seen = 0, set()
            got = False
            for c in o.outputs:
                txt = c.text.strip()
                if extract(txt) is None or extract(txt) != gold:
                    continue
                got = True
                n_correct += 1
                body = txt.rsplit("ANSWER:", 1)[0].strip()
                sig = tuple(sorted(re.findall(r"[-+*/=]\s*-?[\d.]+", body)))
                if not body or sig in seen or kept >= args.keep_per_problem:
                    continue
                seen.add(sig)
                kept += 1
                n_rows += 1
                f.write(json.dumps({"problem": p["problem"], "body": body,
                                    "answer": p["answer"], "src": p.get("src", "rft")}) + "\n")
            solved += int(got)
            stats.append({"src": p.get("src", "?"),
                          "pass": sum(1 for c in o.outputs
                                      if extract(c.text.strip()) == gold),
                          "k": args.k})
    with open(args.out + ".stats.jsonl", "w") as f:
        for r in stats:
            f.write(json.dumps(r) + "\n")
    print(f"[gen] {n_correct} correct samples / {len(probs)*args.k}; "
          f"{solved}/{len(probs)} problems solved at least once; wrote {n_rows} rows",
          flush=True)


if __name__ == "__main__":
    main()
