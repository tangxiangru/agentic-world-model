"""Sample k solutions per question from a checkpoint with vLLM and keep the
correct ones (rejection-sampling fine-tuning / STaR).

Questions come from the GSM8K TRAIN split and from OpenMathInstruct-2's
gsm8k-sourced problems (also train-derived). The GSM8K test split is never
read here.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, "/home/ben/task/scripts")
import render  # noqa: E402

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract(text):
    i = text.rfind("ANSWER:")
    if i == -1:
        return None
    line = text[i + len("ANSWER:") :].split("\n")[0]
    m = NUM.findall(line)
    if not m:
        return None
    v = m[-1].replace(",", "")
    if v.endswith(".0"):
        v = v[:-2]
    return v


def norm(a):
    """Canonical numeric string. Never raises: a model can emit 1e400 or a
    500-digit integer, and an overflow here would throw away a whole sampling
    run at the very last step."""
    a = str(a).strip().replace(",", "").replace("$", "")
    try:
        f = float(a)
        if f != f or f in (float("inf"), float("-inf")):
            return a
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl {question, gold}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-model-len", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    qs = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        qs = qs[: args.limit]
    tok = render.get_tokenizer()
    prompts = [render.render_prompt(tok, q["question"]) for q in qs]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        seed=args.seed,
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    outs = llm.generate(prompts, sp)

    n_corr = n_tot = 0
    per_q = []
    with open(args.out, "w") as fh:
        for q, o in zip(qs, outs):
          try:
            gold = norm(q["gold"])
            sols = []
            for c in o.outputs:
                n_tot += 1
                txt = c.text
                got = extract(txt)
                ok = got is not None and norm(got) == gold
                if ok:
                    n_corr += 1
                    sols.append(txt)
            per_q.append(len(sols))
            fh.write(
                json.dumps(
                    {
                        "question": q["question"],
                        "gold": q["gold"],
                        "n_correct": len(sols),
                        "k": args.k,
                        "solutions": sols,
                    }
                )
                + "\n"
            )
          except Exception as e:  # never lose a finished sampling run
            print("row skipped:", type(e).__name__, e, flush=True)
    solved = sum(1 for x in per_q if x > 0)
    print(
        json.dumps(
            {
                "questions": len(qs),
                "samples": n_tot,
                "correct_samples": n_corr,
                "sample_acc": n_corr / max(1, n_tot),
                "pass_at_k": solved / max(1, len(qs)),
                "unsolved": len(qs) - solved,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
