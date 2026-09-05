#!/usr/bin/env python3
"""Sample k completions per training problem with vLLM and dump everything raw,
so filtering can be re-tuned offline without re-generating."""
import argparse
import json
import random

from datasets import load_from_disk

from prep_sft import render_prompt, norm_answer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n-gsm", type=int, default=7)
ap.add_argument("--n-aug", type=int, default=4)
ap.add_argument("--n-augmented", type=int, default=10000)
ap.add_argument("--temp", type=float, default=0.9)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--fewshot-frac", type=float, default=0.10)
ap.add_argument("--gpu-util", type=float, default=0.88)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--problems-file", default=None,
                help="jsonl of {question, gold, n} overriding the default plan")
ap.add_argument("--skip-questions", default=None,
                help="jsonl of already-covered questions to drop (field 'question')")
args = ap.parse_args()

rng = random.Random(args.seed)
gsm = load_from_disk("data/gsm8k_main")["train"]
gsm_items = []
fewshot_pool = []
for r in gsm:
    parts = r["answer"].split("####")
    gold = norm_answer(parts[-1].strip())
    if gold is not None:
        gsm_items.append((r["question"].strip(), gold))
    fewshot_pool.append(
        f"{r['question']}\n\nReasoning:\n{'####'.join(parts[:-1]).strip()}\n\nANSWER: {parts[-1].strip()}"
    )

aug_items = []
if args.n_augmented > 0:
    omi = load_from_disk("data/omi2_gsm")
    seen = set()
    for r in omi:
        if r["problem_source"] != "augmented_gsm8k":
            continue
        p = r["problem"].strip()
        if p in seen:
            continue
        seen.add(p)
        a = norm_answer(r["expected_answer"])
        if a is not None:
            aug_items.append((p, a))
    rng.shuffle(aug_items)
    aug_items = aug_items[: args.n_augmented]

skip = set()
if args.skip_questions:
    with open(args.skip_questions) as f:
        for line in f:
            skip.add(json.loads(line)["question"].strip())

plan = [(q, g, args.n_gsm) for q, g in gsm_items if q not in skip]
plan += [(q, g, args.n_aug) for q, g in aug_items if q not in skip]
if args.problems_file:
    plan = []
    with open(args.problems_file) as f:
        for line in f:
            r = json.loads(line)
            plan.append((r["question"], r["gold"], r["n"]))
print(f"{len(plan)} problems, {sum(p[2] for p in plan)} completions", flush=True)

prompts, meta = [], []
for q, g, n in plan:
    system = None
    if rng.random() < args.fewshot_frac:
        system = "\n\n".join(rng.sample(fewshot_pool, rng.choice([1, 2, 3])))
    prompts.append((render_prompt(q, system), n))
    meta.append((q, g, system))

from vllm import LLM, SamplingParams

llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=1600,
          dtype="bfloat16", enable_prefix_caching=True, seed=args.seed)
sps = [SamplingParams(n=n, temperature=args.temp, top_p=args.top_p,
                      max_tokens=args.max_tokens, stop_token_ids=[106]) for _, n in prompts]
outs = llm.generate([p for p, _ in prompts], sps)

with open(args.out, "w") as f:
    for out, (q, g, system) in zip(outs, meta):
        f.write(json.dumps({
            "question": q,
            "gold": g,
            "system": system,
            "gens": [{"text": c.text, "fr": c.finish_reason} for c in out.outputs],
        }) + "\n")
print("wrote", args.out)
