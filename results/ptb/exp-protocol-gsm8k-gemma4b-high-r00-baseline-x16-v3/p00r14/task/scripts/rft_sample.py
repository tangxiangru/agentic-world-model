"""Rejection-sampling data: sample k solutions per question, keep the correct ones.

Sampling matches how the grader decodes (the model's own generation_config:
top_k 64, top_p 0.95, temperature 1.0), so the kept solutions are drawn from the
distribution the benchmark will actually sample from. Correctness is decided
with inspect's own match_str against the gold answer.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def degenerate(text):
    """Reject run-on / looping completions that happen to end on a right number."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) > 4 and len(set(lines)) < len(lines) * 0.5:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl of {id, question, gold}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from inspect_ai.scorer._match import match_str
    from fmt import render_prompt, ANSWER_MARKER

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        rows = rows[: args.limit]
    prompts = [render_prompt(tokenizer, r["question"], None) for r in rows]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=2048, seed=args.seed)
    # stop_token_ids, not the stop string: with n>1 vLLM 0.11 stops on the string
    # for only ~12% of children and lets the rest run to max_tokens, which both
    # quadruples the sampling cost and drops per-completion accuracy 0.59 -> 0.33.
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_tokens,
                        stop=["<end_of_turn>"], stop_token_ids=[eot],
                        seed=args.seed)
    outs = llm.generate(prompts, sp)

    kept, per_q_solved, n_any = [], defaultdict(int), 0
    for r, o in zip(rows, outs):
        seen, good = set(), []
        for c in o.outputs:
            t = c.text.strip()
            if c.finish_reason == "length":
                continue
            if not match_str(value=t, target=r["gold"], location="end", numeric=True)[1]:
                continue
            if degenerate(t):
                continue
            if len(re.findall(re.escape(ANSWER_MARKER), t)) != 1:
                continue
            key = re.sub(r"\s+", " ", t)
            if key in seen:
                continue
            seen.add(key)
            good.append(t)
        per_q_solved[r["id"]] = len(good)
        n_any += int(bool(good))
        good.sort(key=len)  # prefer the shorter correct chains
        for t in good[: args.keep_per_question]:
            kept.append({"question": r["question"], "target": t, "answer": r["gold"],
                         "src": "rft:" + r["id"].split("-")[0]})

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    stats = {"questions": len(rows), "k": args.k, "solved_at_least_once": n_any,
             "pass_at_k": n_any / len(rows), "kept_rows": len(kept),
             "unsolved": [q for q, c in per_q_solved.items() if c == 0]}
    if args.stats_out:
        with open(args.stats_out, "w") as f:
            json.dump(stats, f)
    print(json.dumps({k: v for k, v in stats.items() if k != "unsolved"}), flush=True)


if __name__ == "__main__":
    main()
