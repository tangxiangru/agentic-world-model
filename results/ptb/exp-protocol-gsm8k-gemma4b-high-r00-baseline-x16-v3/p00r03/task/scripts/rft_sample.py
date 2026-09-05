"""Rejection sampling: draw k solutions per question from a checkpoint, keep the correct ones.

Questions come from the GSM8K *train* split and from OpenMathInstruct-2's
augmented GSM8K problems (both allowed sources); dev500 items are excluded.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    GEMMA_TEMPLATE,
    TASK_DIR,
    grade,
    gsm8k_gold,
    user_prompt,
    write_jsonl,
)

EOT = "<end_of_turn>"


def norm_answer(a):
    a = str(a).strip().replace(",", "").replace("$", "").rstrip(".")
    return a[:-2] if a.endswith(".0") else a


def collect_questions(n_gsm, n_omi, seed=0):
    from datasets import load_dataset

    dev_idx = set(json.load(open(os.path.join(TASK_DIR, "data", "dev500_train_idx.json"))))
    dev_qs = set(json.load(open(os.path.join(TASK_DIR, "data", "dev500_questions.json"))))
    rng = random.Random(seed)

    out = []
    g = load_dataset("openai/gsm8k", "main")["train"]
    idx = [i for i in range(len(g)) if i not in dev_idx]
    rng.shuffle(idx)
    for i in idx[:n_gsm]:
        r = g[i]
        out.append({"id": f"gsm-{i}", "question": r["question"], "gold": norm_answer(gsm8k_gold(r["answer"]))})

    if n_omi > 0:
        omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
        srcs = omi["problem_source"]
        cand = [i for i, s in enumerate(srcs) if s in ("gsm8k", "augmented_gsm8k")]
        rng.shuffle(cand)
        seen = set()
        for i in cand:
            if len(seen) >= n_omi:
                break
            r = omi[i]
            q = r["problem"]
            if q in dev_qs or q in seen:
                continue
            a = norm_answer(r["expected_answer"])
            if not re.match(r"^-?\d+$", a):
                continue
            seen.add(q)
            out.append({"id": f"omi-{i}", "question": q, "gold": a})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-out", default=None)
    ap.add_argument("--n-gsm", type=int, default=6973)
    ap.add_argument("--n-omi", type=int, default=6000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    qs = collect_questions(args.n_gsm, args.n_omi, args.seed)
    print(f"questions: {len(qs)}")

    with open(GEMMA_TEMPLATE) as f:
        chat_template = f.read()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        dtype="bfloat16",
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.k,
        seed=args.seed,
    )
    convs = [[{"role": "user", "content": user_prompt(q["question"])}] for q in qs]
    outs = llm.chat(convs, sp, chat_template=chat_template, add_generation_prompt=True)

    rows, n_solved, n_kept = [], 0, 0
    raw = []
    for q, o in zip(qs, outs):
        texts = [c.text for c in o.outputs]
        good = [t for t in texts if grade(t, q["gold"])]
        raw.append({"id": q["id"], "gold": q["gold"], "n_correct": len(good), "k": len(texts)})
        if good:
            n_solved += 1
        # Questions the model already solves every time teach it little, so keep
        # one solution from those and up to keep_per_question from the rest.
        cap = 1 if len(good) == len(texts) else args.keep_per_question
        seen, chosen = set(), []
        for t in sorted(good, key=len):
            key = re.sub(r"\s+", " ", t).strip()
            if key in seen:
                continue
            seen.add(key)
            chosen.append(t)
            if len(chosen) >= cap:
                break
        for t in chosen:
            rows.append(
                {
                    "prompt": user_prompt(q["question"]),
                    "completion": t.strip() + EOT,
                    "question": q["question"],
                    "answer": q["gold"],
                    "source": "rft_self",
                }
            )
            n_kept += 1

    print(f"solved at least once: {n_solved}/{len(qs)} = {n_solved/len(qs):.3f}")
    print(f"kept rows: {n_kept}")
    write_jsonl(args.out, rows)
    if args.raw_out:
        with open(args.raw_out, "w") as f:
            json.dump(raw, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
