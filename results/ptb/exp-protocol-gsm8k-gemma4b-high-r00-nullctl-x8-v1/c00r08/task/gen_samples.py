"""Offline vLLM sampling over math problems: quick local eval + rejection-sampling data.

Modes:
  --mode eval  : k=1 greedy (or sampled) pass over a question set, print accuracy.
  --mode rft   : k samples/question at temperature T, keep the ones whose final
                 number matches the gold answer, write an SFT-ready JSONL.
"""
import argparse
import json
import os
import random

from transformers import AutoTokenizer

from common import (
    BASE_SNAPSHOT,
    extract_answer,
    gsm8k_fewshots,
    load_chat_template,
    normalize_num,
    user_prompt,
)


def load_questions(args):
    from datasets import load_dataset

    rows = []
    if args.source == "gsm8k_train":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        for r in ds:
            a = normalize_num(r["answer"].split("####")[-1].strip())
            if a is not None:
                rows.append({"question": r["question"].strip(), "answer": a})
    else:  # jsonl pool file
        seen = set()
        for line in open(args.source):
            r = json.loads(line)
            q = r["question"].strip()
            if q in seen:
                continue
            seen.add(q)
            rows.append({"question": q, "answer": r["answer"]})
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.start:
        rows = rows[args.start :]
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["eval", "rft"], default="eval")
    ap.add_argument("--source", default="gsm8k_train")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--fewshot", type=int, default=0, help="use the eval 10-shot system prompt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-per-q", type=int, default=4)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE_SNAPSHOT)
    tok.chat_template = load_chat_template()

    rows = load_questions(args)
    print(f"questions: {len(rows)}")

    system = "\n\n".join(gsm8k_fewshots(10, seed=42, shuffle=True)) if args.fewshot else None
    prompts = []
    for r in rows:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user_prompt(r["question"])}
        ]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_util,
        max_model_len=4096,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
    )
    outs = llm.generate(prompts, sp)

    if args.out:  # checkpoint raw generations before scoring
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out + ".raw", "w") as f:
            for r, o in zip(rows, outs):
                f.write(json.dumps({**r, "gens": [c.text for c in o.outputs]}) + "\n")

    n_correct = 0
    kept = []
    per_q_pass = []
    for r, o in zip(rows, outs):
        texts = [c.text for c in o.outputs]
        oks = [extract_answer(t) == r["answer"] for t in texts]
        per_q_pass.append(sum(oks) / len(oks))
        n_correct += oks[0]
        if args.mode == "rft":
            good = [t.strip() for t, ok in zip(texts, oks) if ok]
            # dedupe near-identical solutions, prefer shorter ones
            uniq, seen = [], set()
            for t in sorted(good, key=len):
                key = tuple(sorted(set(w for w in t.split() if any(ch.isdigit() for ch in w))))
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(t)
            for t in uniq[: args.keep_per_q]:
                if not t.rstrip().endswith(r["answer"]):
                    continue
                kept.append(
                    dict(question=r["question"], solution=t, answer=r["answer"], source="rft")
                )

    print(f"pass@1(first sample): {n_correct/len(rows):.4f}")
    print(f"mean pass rate over k={args.k}: {sum(per_q_pass)/len(per_q_pass):.4f}")
    print(f"solved at least once: {sum(1 for p in per_q_pass if p>0)/len(per_q_pass):.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(kept)} rows -> {args.out}")
        stats = os.path.splitext(args.out)[0] + "_qstats.json"
        with open(stats, "w") as f:
            json.dump(
                [{"question": r["question"], "answer": r["answer"], "pass": p}
                 for r, p in zip(rows, per_q_pass)], f)


if __name__ == "__main__":
    main()
