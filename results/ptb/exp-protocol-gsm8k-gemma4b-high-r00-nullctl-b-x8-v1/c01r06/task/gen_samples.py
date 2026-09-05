"""Sample solutions for GSM8K *train* questions from an SFT checkpoint (rejection sampling)."""
import argparse
import json
import os
import random

from datasets import load_dataset
from vllm import LLM, SamplingParams

from common import (extract_answer, fewshot_block, norm_num, render_prompt,
                    split_gsm8k_answer)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=640)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--qidx-file", default=None,
                   help="JSON list of GSM8K-train indices to restrict sampling to")
    p.add_argument("--fewshot-frac", type=float, default=0.15)
    p.add_argument("--gpu-util", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse()
    rng = random.Random(args.seed)

    g = load_dataset("openai/gsm8k", "main", split="train")
    items = []
    demos = []
    for q, a in zip(g["question"], g["answer"]):
        reasoning, target = split_gsm8k_answer(a)
        demos.append(fewshot_block(q, reasoning, target))
        items.append({"question": q.strip(), "answer": norm_num(target)})
    if args.limit > 0:
        items = items[: args.limit]

    wanted = None
    if args.qidx_file:
        wanted = set(json.load(open(args.qidx_file)))
        print(f"restricting to {len(wanted)} questions")

    prompts, meta = [], []
    for idx, it in enumerate(items):
        if wanted is not None and idx not in wanted:
            continue
        for j in range(args.k):
            prefix = ""
            if rng.random() < args.fewshot_frac:
                prefix = "\n\n".join(rng.sample(demos, rng.randint(3, 10)))
            prompts.append(render_prompt(it["question"], prefix))
            meta.append(idx)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util,
              max_model_len=3072, dtype="bfloat16", enforce_eager=False,
              disable_log_stats=True)
    sp = SamplingParams(n=1, temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=None,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    n_ok = 0
    with open(args.out, "w") as f:
        for m, o in zip(meta, outs):
            text = o.outputs[0].text.strip()
            pred = norm_num(extract_answer(text))
            ok = pred is not None and pred == items[m]["answer"]
            n_ok += ok
            f.write(json.dumps({"qidx": m, "question": items[m]["question"],
                                "answer": items[m]["answer"], "completion": text,
                                "correct": bool(ok)}) + "\n")
    print(f"generated {len(outs)} samples, {n_ok} correct ({n_ok/len(outs):.3f})")


if __name__ == "__main__":
    main()
