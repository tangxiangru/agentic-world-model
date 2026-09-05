#!/usr/bin/env python3
"""Generate + score completions with vLLM using the exact inspect-eval prompt format."""
import argparse
import json
import os
import random

from common import build_messages, eval_fewshot_system, extract_pred, get_tokenizer, norm_num


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="jsonl with question/answer")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--fewshot", default="eval", choices=["eval", "none", "random"])
    ap.add_argument("--fewshot-seed", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.data)]
    if args.limit > 0:
        items = items[: args.limit]

    tok = get_tokenizer()
    sysmsg = None
    if args.fewshot == "eval":
        sysmsg = eval_fewshot_system(10, 42)

    rng = random.Random(args.fewshot_seed)
    pool = None
    if args.fewshot == "random":
        from datasets import load_dataset
        devq = set()
        if os.path.exists("data/dev.jsonl"):
            devq = {json.loads(l)["question"] for l in open("data/dev.jsonl")}
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        pool = []
        for rec in gsm:
            if rec["question"].strip() in devq:
                continue
            body, tgt = rec["answer"].split("####")
            pool.append(f"{rec['question'].strip()}\n\nReasoning:\n{body.strip()}\n\nANSWER: {tgt.strip()}")

    prompts = []
    for it in items:
        s = sysmsg
        if pool is not None:
            k = rng.choice([0, 2, 4, 6, 8, 10])
            s = "\n\n".join(rng.sample(pool, k)) if k else None
        msgs = build_messages(it["question"], s)
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=4096,
        enforce_eager=False,
        seed=args.seed,
        disable_log_stats=True,
    )
    sp = SamplingParams(
        n=args.n,
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<end_of_turn>", "<start_of_turn>"],
        seed=None,
    )
    outs = llm.generate(prompts, sp)

    n_correct = 0
    n_any = 0
    recs = []
    for it, o in zip(items, outs):
        gold = norm_num(it["answer"])
        cands = []
        for c in o.outputs:
            txt = c.text
            pred = extract_pred(txt)
            ok = pred is not None and pred == gold
            cands.append({"text": txt, "pred": pred, "correct": ok})
        n_correct += sum(c["correct"] for c in cands) / len(cands)
        n_any += int(any(c["correct"] for c in cands))
        recs.append({"question": it["question"], "answer": gold, "cands": cands})

    print(f"MEAN_ACC {n_correct/len(items):.4f}  PASS@{args.n} {n_any/len(items):.4f}  N={len(items)}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
