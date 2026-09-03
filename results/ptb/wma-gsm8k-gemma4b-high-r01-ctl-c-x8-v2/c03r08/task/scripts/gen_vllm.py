#!/usr/bin/env python3
"""Generate + score completions with vLLM, using the grader's exact rendering.

Serves two jobs:
  * probe evaluation on held-out gsm8k TRAIN questions (data/probe250.jsonl),
    which is the only place failure analysis is allowed to look (rule 7);
  * rejection sampling for RFT: --n k --temperature t, keep the correct ones.

Scoring reuses inspect_ai's own match_str(location="end", numeric=True), i.e.
the last number-like token in the completion, so a probe number means the same
thing the benchmark number means.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_ai.scorer._common import match_str
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ROOT = Path("/home/ben/task")

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def is_correct(completion: str, gold: str) -> bool:
    _, ok = match_str(value=completion, target=gold, location="end",
                      ignore_case=True, numeric=True)
    return bool(ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", default=str(ROOT / "data" / "probe250.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--fewshot", action="store_true",
                    help="prepend the grader's exact 10-shot system message")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    template = (ROOT / "templates" / "gemma3.jinja").read_text()
    tok = AutoTokenizer.from_pretrained(args.model)
    items = [json.loads(l) for l in Path(args.questions).open()]
    if args.limit:
        items = items[: args.limit]

    fs = (ROOT / "data" / "fewshot_prefix.txt").read_text() if args.fewshot else None
    prompts = []
    for it in items:
        msgs = ([{"role": "system", "content": fs}] if fs else []) + [
            {"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=it["question"].strip())}
        ]
        prompts.append(tok.apply_chat_template(
            msgs, chat_template=template, tokenize=False, add_generation_prompt=True))
    print(f"{len(prompts)} prompts; example head:\n{prompts[0][:300]}\n...")

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, dtype="bfloat16", seed=args.seed,
              enforce_eager=False)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_tokens,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    n_any = n_first = n_tot = 0
    with Path(args.out).open("w") as f:
        for it, o in zip(items, outs):
            texts = [c.text for c in o.outputs]
            flags = [is_correct(t, str(it["gold"])) for t in texts]
            n_tot += 1
            n_first += flags[0]
            n_any += any(flags)
            f.write(json.dumps({"id": it.get("id"), "question": it["question"],
                                "gold": str(it["gold"]), "samples": texts,
                                "correct": flags}) + "\n")
    print(json.dumps({"n": n_tot, "acc_first_sample": round(n_first / n_tot, 4),
                      f"pass@{args.n}": round(n_any / n_tot, 4)}, indent=1))


if __name__ == "__main__":
    main()
