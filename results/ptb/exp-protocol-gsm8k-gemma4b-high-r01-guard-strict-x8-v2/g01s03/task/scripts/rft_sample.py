#!/usr/bin/env python3
"""Rejection-sampling data: sample k solutions per gsm8k TRAIN question from a
checkpoint, keep the ones whose ANSWER line matches the gold answer.

Only the gsm8k train split is used as the question source; the test split is
never read here.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

ANS_RE = re.compile(r"ANSWER:\s*([-$]?[\d,]*\.?\d+)\s*$")


def norm(s):
    s = s.strip().replace(",", "").replace("$", "")
    try:
        f = float(s)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else ("%.6f" % f).rstrip("0").rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--n-questions", type=int, default=7473)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.2)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    tok = fmt.get_tokenizer()
    system = fmt.fewshot_system_message()

    ds = load_dataset("openai/gsm8k", "main", split="train")
    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)
    idx = idx[: args.n_questions]
    qs = [ds[i]["question"] for i in idx]
    golds = [norm(ds[i]["answer"].rsplit("####", 1)[1]) for i in idx]

    prompts = [fmt.render_prompt(tok, q, None) for q in qs]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
        enable_prefix_caching=True,
        dtype="bfloat16",
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed + 1)
    kept = []
    per_q_correct = []
    for q, gold, out in zip(qs, golds, outs):
        cands = []
        for o in out.outputs:
            text = o.text.strip()
            m = ANS_RE.search(text)
            if not m or norm(m.group(1)) != gold:
                continue
            body = text[: m.start()].strip()
            if not body:
                continue
            cands.append(body)
        per_q_correct.append(len(cands))
        # prefer short, distinct solutions
        seen = set()
        uniq = []
        for c in sorted(cands, key=len):
            key = re.sub(r"\s+", " ", c)[:200]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        for body in uniq[: args.max_per_question]:
            use_fs = rng.random() < args.fewshot_frac
            kept.append({
                "question": q,
                "answer": body + "\nANSWER: " + gold,
                "prompt": fmt.render_prompt(tok, q, system if use_fs else None),
                "completion": fmt.render_target(body, gold),
                "source": "rft",
                "fewshot": int(use_fs),
            })

    solved = sum(1 for c in per_q_correct if c > 0)
    print(f"questions {len(qs)}  solved_at_least_once {solved} "
          f"({solved/len(qs):.3f})  pass@1 {sum(per_q_correct)/(len(qs)*args.k):.3f}  "
          f"kept_rows {len(kept)}", flush=True)
    rng.shuffle(kept)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print("wrote", args.out, flush=True)
    with open(args.out + ".stats.json", "w") as f:
        json.dump({"n_questions": len(qs), "k": args.k,
                   "solved_at_least_once": solved,
                   "pass_at_1": sum(per_q_correct) / (len(qs) * args.k),
                   "kept_rows": len(kept)}, f, indent=2)


if __name__ == "__main__":
    main()
