"""Rejection-sampling data generation: sample k solutions per training problem
from a fine-tuned checkpoint, keep the ones whose final number matches gold.

Problems come from data/gsm8k_train_pool.jsonl (GSM8K *train*, probe hold-outs
already removed) and optionally from the OpenMathInstruct-2 augmented problems,
whose gold answers the dataset ships.  No benchmark test item is involved.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import fmt as F  # noqa: E402


def final_number(text: str) -> str | None:
    words = re.split(r"\s+", text.strip())[::-1]
    for w in words:
        w2 = w.strip().strip(".,:;!?$()[]%").replace(",", "")
        if w2.replace(".", "").replace("-", "").isnumeric():
            return w2.rstrip(".").rstrip("0").rstrip(".") if "." in w2 else w2
    return None


def norm(a: str) -> str:
    a = a.strip().replace(",", "")
    if "." in a:
        a = a.rstrip("0").rstrip(".")
    return a


def equation_signature(body: str) -> str:
    """Cheap dedup key: the sequence of numbers the solution produces."""
    return "|".join(re.findall(r"-?\d+(?:\.\d+)?", body))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--problems", default="data/gsm8k_train_pool.jsonl")
    ap.add_argument("--extra-problems", default=None,
                    help="jsonl with problem/expected_answer (OpenMathInstruct-2 rows)")
    ap.add_argument("--extra-limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    probs = []
    seen_q = set()
    for line in open(args.problems):
        r = json.loads(line)
        gold = norm(r["answer"].rsplit("####", 1)[-1])
        probs.append({"question": r["question"], "gold": gold, "src": "gsm8k_train"})
        seen_q.add(r["question"])
    if args.extra_problems and args.extra_limit:
        per = {}
        for line in open(args.extra_problems):
            r = json.loads(line)
            q = r["problem"]
            if q in seen_q or q in per:
                continue
            g = norm(r["expected_answer"])
            if not g.replace(".", "").replace("-", "").isnumeric():
                continue
            per[q] = g
            if len(per) >= args.extra_limit:
                break
        probs += [{"question": q, "gold": g, "src": "omi2_problem"} for q, g in per.items()]
    print(f"{len(probs)} problems, k={args.k}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model_path)
    prompts = [F.render_prompt(tok, p["question"], fewshot=bool(args.fewshot)) for p in probs]
    llm = LLM(model=args.model_path, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=3072, dtype="bfloat16")
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=0)
    outs = llm.generate(prompts, sp)

    kept, n_corr, n_tot = [], 0, 0
    solved = 0
    per_problem_correct = []
    for p, o in zip(probs, outs):
        sigs = set()
        this = []
        n_ok_here = 0
        for c in o.outputs:
            n_tot += 1
            txt = c.text.strip()
            if final_number(txt) != p["gold"]:
                continue
            n_ok_here += 1
            n_corr += 1
            if txt.count(F.ANSWER_MARKER) != 1:
                continue
            body = txt.split(F.ANSWER_MARKER)[0].strip()
            if len(body) < 20:
                continue
            sig = equation_signature(body)
            if sig in sigs:
                continue
            sigs.add(sig)
            this.append({"question": p["question"], "body": body,
                         "answer": p["gold"], "src": "rft_" + p["src"]})
        per_problem_correct.append(n_ok_here)
        solved += int(n_ok_here > 0)
        kept += this[: args.keep_per_problem]

    stats = {"n_problems": len(probs), "k": args.k, "n_samples": n_tot,
             "sample_accuracy": n_corr / max(1, n_tot),
             "problems_solved_at_least_once": solved,
             "pass_at_k": solved / max(1, len(probs)),
             "kept_rows": len(kept)}
    print(json.dumps(stats, indent=1), flush=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    if args.stats_out:
        json.dump({"stats": stats,
                   "per_problem_correct": per_problem_correct},
                  open(args.stats_out, "w"), indent=1)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
