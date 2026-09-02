#!/usr/bin/env python3
"""Rejection-sampling data build: draw k solutions per training question from a
checkpoint, keep the ones whose final ANSWER line matches the gold answer.

Prompts are rendered exactly as the grader renders them (templates/gemma3.jinja,
single user turn), so the samples come from the same distribution the model is
graded in.
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict

STOP = "<end_of_turn>"
ANSWER_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def norm(a: str) -> str:
    """Normalise a numeric answer for comparison. Never raises: a sampled
    'ANSWER: 999999999999999999999999' parses to a float that overflows int(),
    and one such sample killed an 80-minute generation run."""
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if len(a) > 24:
        return a
    try:
        f = float(a)
        if f != f or f in (float("inf"), float("-inf")):
            return a
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        return a


def load_questions(path: str, limit: int | None, seed: int):
    """Prompt pool built by scripts/build_questions.py: every distinct GSM8K-family
    training problem exp-02's corpus was drawn from (gsm8k train +
    OpenMathInstruct-2 gsm8k/augmented_gsm8k). No new provenance."""
    out = [json.loads(l) for l in open(path)]
    random.Random(seed).shuffle(out)
    return out[:limit] if limit else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", default="/home/ben/task/data/questions_pool.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--decon-out", default=None)
    ap.add_argument("--n-questions", type=int, default=20000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    qs = load_questions(args.src, args.n_questions, args.seed)
    print(f"questions: {len(qs)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tpl = open("/home/ben/task/templates/gemma3.jinja").read()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q["question"])}],
            chat_template=tpl, tokenize=False, add_generation_prompt=True,
        )
        for q in qs
    ]

    # gemma-3-4b decodes slowly (262k vocab), so throughput here is bought with
    # concurrency: a 1024-token window still fits every prompt+completion and lets
    # the same KV cache hold ~4x more sequences than the 4096 default.
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
              dtype="bfloat16", seed=args.seed, enable_prefix_caching=True)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed + 7)
    kept = 0
    per_q = defaultdict(list)
    n_any = n_all = 0
    for q, o in zip(qs, outs):
        good = []
        for c in o.outputs:
            txt = c.text.strip()
            m = ANSWER_RE.search(txt)
            if not m or norm(m.group(1)) != q["gold"]:
                continue
            if txt.count("ANSWER:") != 1:
                continue
            good.append(txt)
        if good:
            n_any += 1
        mastered = len(good) == args.k
        n_all += mastered
        # Concentrate the corpus on the frontier: a problem the model already
        # solves on every draw teaches it almost nothing, so it contributes one
        # solution; a problem it solves only sometimes contributes two.
        cap = 1 if mastered else args.max_per_question
        uniq = sorted(set(good))
        rng.shuffle(uniq)
        good = uniq[:cap]
        per_q[q["question"]] = (good, q["gold"])
        kept += len(good)

    print(f"questions with >=1 correct sample: {n_any}/{len(qs)} "
          f"({n_any/len(qs):.3f}); solved on every draw: {n_all} "
          f"({n_all/len(qs):.3f}); kept solutions: {kept}")

    dec = open(args.decon_out, "w") if args.decon_out else None
    with open(args.out, "w") as f:
        for q, (sols, gold) in per_q.items():
            for s in sols:
                f.write(json.dumps({
                    "prompt": MATH_PROMPT_TEMPLATE.format(prompt=q),
                    "completion": s + STOP,
                    "src": "rft_self",
                }) + "\n")
                if dec:
                    dec.write(json.dumps({"question": q, "answer": s}) + "\n")
    if dec:
        dec.close()
    print(f"wrote {kept} rows -> {args.out}")
    json.dump({"n_questions": len(qs), "solve_rate_any_of_k": n_any / len(qs),
               "solved_all_k": n_all / len(qs),
               "kept": kept, "k": args.k, "temperature": args.temperature},
              open(args.out + ".stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
