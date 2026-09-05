#!/usr/bin/env python3
"""Offline vLLM helper: dev-set scoring and rejection-sampling generation.

Prompts are rendered exactly as templates/gemma3.jinja renders them, and the
answer is read the way inspect_ai's match(location="end", numeric=True) reads
it: the last number in the completion.
"""
from __future__ import annotations

import argparse
import json
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def render(user_text: str) -> str:
    return (
        "<bos><start_of_turn>user\n"
        + user_text.strip()
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def last_number(text: str) -> str | None:
    t = text.replace(",", "")
    m = NUM.findall(t)
    return m[-1] if m else None


def eq(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def fewshot_prefix(n: int, seed: int = 42) -> str:
    """A 10-shot block in the harness's own format, from the GSM8K TRAIN split."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    parts = []
    for r in ds.select(range(n)):
        body, _, ans = r["answer"].partition("####")
        parts.append(
            f"{r['question']}\n\nReasoning:\n{body.strip()}\n\nANSWER: {ans.strip()}"
        )
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["eval", "sample"], default="eval")
    ap.add_argument("--questions", default="data/dev_train300.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        rows = rows[: args.limit]

    prefix = fewshot_prefix(args.fewshot) + "\n\n" if args.fewshot else ""

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for r in rows:
        user = prefix + MATH_PROMPT_TEMPLATE.format(prompt=r["question"].strip())
        ids = tok(render(user), add_special_tokens=False)["input_ids"]
        prompts.append(TokensPrompt(prompt_token_ids=ids))

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
        enforce_eager=False,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
    )
    outs = llm.generate(prompts, sp)

    n_ok = 0
    n_fmt = 0
    recs = []
    for r, o in zip(rows, outs):
        gold = str(r["gold"])
        cands = []
        for c in o.outputs:
            text = c.text.strip()
            pred = last_number(text)
            ok = eq(pred, gold)
            fmt = bool(re.search(r"ANSWER:\s*-?[\d,\.]+\s*$", text))
            cands.append({"text": text, "pred": pred, "correct": ok, "fmt_ok": fmt})
        n_ok += int(cands[0]["correct"])
        n_fmt += int(cands[0]["fmt_ok"])
        recs.append({"id": r.get("id"), "question": r["question"], "gold": gold,
                     "cands": cands})

    n = len(rows)
    summary = {
        "model": args.model,
        "mode": args.mode,
        "n": n,
        "k": args.k,
        "temperature": args.temperature,
        "fewshot": args.fewshot,
        "acc_first_sample": n_ok / n,
        "fmt_ok_first_sample": n_fmt / n,
        "pass_at_k": sum(any(c["correct"] for c in r["cands"]) for r in recs) / n,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "records": recs}, f)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
