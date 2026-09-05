#!/usr/bin/env python3
"""Build self-consistency-in-one-pass training data from my own SFT checkpoint.

For each problem, sample k completions, then assemble a single target that
contains three of those attempts followed by an explicit majority vote:

    Attempt 1:
    <solution 1>
    So the answer is 42.

    Attempt 2:
    ...

    The three attempts give 42, 37, 42. The majority answer is 42.

    ANSWER: 42<end_of_turn>

Only problems whose chosen triple has a correct majority are kept, and where
possible the triple mixes correct and incorrect attempts so the vote is doing
real work rather than rubber-stamping three identical derivations.

Gold answers come from GSM8K-train-derived OpenMathInstruct-2 problems; no test
item is read.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

from build_data import MATH_PROMPT_TEMPLATE, STOP, render, sample_to_fewshot
from dev_eval import graded_answer, norm

ANS_LINE = re.compile(r"\n*ANSWER:[^\n]*\n?", re.IGNORECASE)


def strip_answer_line(text: str) -> str:
    return ANS_LINE.sub("", text).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sources", nargs="+", default=["data/sft_omi2_gsm8k.jsonl"])
    ap.add_argument("--n-problems", type=int, default=12000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    seen, problems = set(), []
    for src in args.sources:
        with open(src) as f:
            for line in f:
                r = json.loads(line)
                q = r["question"]
                if q in seen:
                    continue
                m = re.search(r"ANSWER: ([^\n<]+)", r["completion"])
                if not m:
                    continue
                seen.add(q)
                problems.append((q, m.group(1).strip()))
    rng.shuffle(problems)
    problems = problems[: args.n_problems]
    print(f"sampling k={args.k} for {len(problems)} problems", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=1024, dtype="bfloat16")
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    prompts = [render(None, MATH_PROMPT_TEMPLATE.format(prompt=q)) for q, _ in problems]
    outs = llm.generate(prompts, sp)

    from datasets import load_dataset

    train = load_dataset("openai/gsm8k", "main", split="train")
    demos = []
    for r in train:
        parts = r["answer"].split("####")
        demos.append((r["question"], "####".join(parts[:-1]).strip(), parts[-1].strip()))

    hist = Counter()
    kind = Counter()
    n_written = 0
    with open(args.out, "w") as f:
        for (q, gold), o in zip(problems, outs):
            g = norm(gold)
            good, bad = [], []
            for c in o.outputs:
                if c.finish_reason == "length":
                    continue
                a = graded_answer(c.text)
                if a is None:
                    continue
                body = strip_answer_line(c.text)
                if not body or "ANSWER" in body.upper():
                    continue
                (good if a == g else bad).append((body, a))
            hist[len(good)] += 1
            # dedup by body
            def dedup(xs):
                out, s = [], set()
                for b, a in xs:
                    if b not in s:
                        s.add(b)
                        out.append((b, a))
                return out

            good, bad = dedup(good), dedup(bad)
            if len(good) < 2:
                continue
            if bad:
                triple = good[:2] + bad[:1]
                kind["mixed"] += 1
            elif len(good) >= 3:
                triple = good[:3]
                kind["unanimous"] += 1
            else:
                continue
            rng.shuffle(triple)

            parts = []
            for i, (body, a) in enumerate(triple, 1):
                num = a.rstrip("0").rstrip(".") if "." in a else a
                parts.append(f"Attempt {i}:\n{body}\nSo the answer is {num}.")
            nums = [a.rstrip("0").rstrip(".") if "." in a else a for _, a in triple]
            maj = Counter(nums).most_common(1)[0][0]
            body = "\n\n".join(parts)
            body += (
                f"\n\nThe three attempts give {nums[0]}, {nums[1]}, {nums[2]}. "
                f"The majority answer is {maj}."
            )
            comp = f"{body}\n\nANSWER: {maj}{STOP}"
            if comp.count("ANSWER: ") != 1:
                continue

            system = None
            if rng.random() < args.fewshot_frac:
                k = rng.randint(2, 10)
                system = "\n\n".join(sample_to_fewshot(*d) for d in rng.sample(demos, k))
            f.write(
                json.dumps(
                    {
                        "prompt": render(system, MATH_PROMPT_TEMPLATE.format(prompt=q)),
                        "completion": comp,
                        "question": q,
                        "answer": body,
                    }
                )
                + "\n"
            )
            n_written += 1

    stats = {
        "problems": len(problems),
        "k": args.k,
        "n_correct_histogram": {str(i): hist[i] for i in range(args.k + 1)},
        "triples": dict(kind),
        "rows_written": n_written,
    }
    print(json.dumps(stats, indent=1))
    if args.stats_out:
        json.dump(stats, open(args.stats_out, "w"), indent=1)


if __name__ == "__main__":
    main()
