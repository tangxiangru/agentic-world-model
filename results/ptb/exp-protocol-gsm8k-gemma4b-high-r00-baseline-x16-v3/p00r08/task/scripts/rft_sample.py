#!/usr/bin/env python3
"""Rejection-sampling data: draw k solutions per training question from a
checkpoint, keep the ones whose ANSWER line matches the gold answer.

Prompts are rendered exactly as the grader renders them (common.render_prompt),
so the samples are on-policy for the evaluation condition.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with prompt_question/answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-questions", type=int, default=0)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--fewshot-p", type=float, default=0.2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.questions)]
    # one entry per distinct question
    byq = {}
    for r in rows:
        byq.setdefault(r["prompt_question"], r["answer"])
    items = [{"q": q, "a": a} for q, a in byq.items()]
    rng = random.Random(args.seed)
    rng.shuffle(items)
    if args.max_questions:
        items = items[: args.max_questions]
    print(f"{len(items)} distinct questions", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    pool = common.eval_fewshot_samples()
    blocks = [common.fewshot_block(s) for s in pool]

    prompts = []
    for it in items:
        system = None
        if rng.random() < args.fewshot_p:
            idx = [i for i in range(len(blocks)) if pool[i].input.strip() != it["q"]]
            system = "\n\n".join(blocks[i] for i in idx)
        prompts.append(common.render_prompt(tok, it["q"], system))

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        enforce_eager=False,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, common.STOP_TOKEN_ID],
        seed=args.seed,
    )
    outs = llm.generate(prompts, sp)

    kept = 0
    n_solved = 0
    seen_bodies = defaultdict(set)
    with open(args.out, "w") as f:
        for it, o in zip(items, outs):
            good = []
            for c in o.outputs:
                text = c.text.strip()
                m = ANS_RE.search(text)
                if not m:
                    continue
                if m.group(1).replace(",", "") != it["a"]:
                    continue
                if text in seen_bodies[it["q"]]:
                    continue
                seen_bodies[it["q"]].add(text)
                good.append(text)
            if good:
                n_solved += 1
            # prefer the shortest correct chains: fewer places to slip
            good.sort(key=len)
            for text in good[: args.keep_per_question]:
                f.write(
                    json.dumps(
                        {
                            "prompt_question": it["q"],
                            "target_reasoning": text.rsplit("ANSWER:", 1)[0].strip(),
                            "answer": it["a"],
                            "source": "rft_self",
                            "target": text + common.STOP_TOKEN,
                        }
                    )
                    + "\n"
                )
                kept += 1
    print(
        f"solved {n_solved}/{len(items)} ({n_solved/len(items):.3f}); kept {kept} rows -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
