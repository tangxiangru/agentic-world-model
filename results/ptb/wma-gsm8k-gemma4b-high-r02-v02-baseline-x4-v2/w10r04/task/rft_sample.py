#!/usr/bin/env python3
"""Rejection-sampling data: draw k samples per question from a trained checkpoint,
keep the ones whose graded last number equals the gold answer.

Questions come from the gsm8k TRAIN split and from OpenMathInstruct-2's
augmented_gsm8k problems (themselves derived from gsm8k train).  The 250
dev-holdout questions are excluded.  Output rows have the same prompt/completion
shape as build_data.py so train_sft.py can read them unchanged.
"""
import argparse
import glob
import json
import random
import re
from collections import defaultdict

from build_data import (EOT, NUM_OK, build_completion, build_prompt,
                        clean_gsm8k_cot, last_number, sample_to_fewshot)


def normalize(s):
    try:
        f = float(str(s).replace(",", ""))
    except ValueError:
        return str(s)
    return str(int(f)) if f == int(f) else f"{f:.5f}".rstrip("0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/rft_v1.jsonl")
    ap.add_argument("--n-aug", type=int, default=9000, help="augmented_gsm8k questions to add")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-out", default="data/rft_v1_stats.json")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    gsm = load_from_disk("data/gsm8k_raw")["train"]
    held = {json.loads(l)["question"].strip() for l in open("data/dev_holdout.jsonl")}

    items = []          # (question, gold)
    for r in gsm:
        if r["question"].strip() in held:
            continue
        _, final = clean_gsm8k_cot(r["answer"])
        items.append((r["question"].strip(), final))
    n_gsm = len(items)

    if a.n_aug:
        import pyarrow.parquet as pq
        seen = set()
        aug = []
        for f in sorted(glob.glob("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
                                  "snapshots/*/data/*.parquet")):
            df = pq.read_table(f).to_pandas()
            sub = df[df.problem_source == "augmented_gsm8k"]
            for q, ans in zip(sub["problem"], sub["expected_answer"]):
                q = q.strip()
                if q in seen or q in held or not NUM_OK.match(str(ans)):
                    continue
                seen.add(q)
                aug.append((q, str(ans)))
            del df
            if len(aug) > a.n_aug * 3:
                break
        rng.shuffle(aug)
        items += aug[: a.n_aug]

    # few-shot pool, same construction as build_data.py
    pool = []
    for r in gsm.shuffle(seed=7).select(range(2500)):
        if r["question"].strip() in held:
            continue
        cot, final = clean_gsm8k_cot(r["answer"])
        pool.append(sample_to_fewshot(r["question"], cot, final))

    tok = AutoTokenizer.from_pretrained(a.model)
    tmpl = open("templates/gemma3.jinja").read()
    prompts = []
    for q, _ in items:
        from build_data import MATH_PROMPT_TEMPLATE
        msgs = [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}]
        prompts.append(tok.apply_chat_template(msgs, chat_template=tmpl, tokenize=False,
                                               add_generation_prompt=True))
    print(f"{len(items)} questions ({n_gsm} gsm8k train + {len(items)-n_gsm} augmented), k={a.k}")

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_memory_utilization,
              max_model_len=4096, dtype="bfloat16")
    sp = SamplingParams(temperature=a.temperature, top_p=0.95, n=a.k,
                        max_tokens=a.max_tokens, stop_token_ids=[1, 106], seed=a.seed)
    outs = llm.generate(prompts, sp)

    rows = []
    n_solved = 0
    per_q_correct = []
    for (q, gold), o in zip(items, outs):
        good, seen_txt = [], set()
        for c in o.outputs:
            t = c.text.strip()
            if c.finish_reason == "length" or "ANSWER:" not in t:
                continue
            if last_number(t) is None or normalize(last_number(t)) != normalize(gold):
                continue
            body = t.rsplit("ANSWER:", 1)[0].strip()
            if not body or len(body) > 2600:
                continue
            key = re.sub(r"\s+", " ", body)[:400]
            if key in seen_txt:
                continue
            seen_txt.add(key)
            good.append(body)
        per_q_correct.append(len(good))
        if good:
            n_solved += 1
        rng.shuffle(good)
        for body in good[: a.keep_per_question]:
            block = None
            if rng.random() < a.fewshot_frac:
                block = "\n\n".join(rng.sample(pool, rng.choice([1, 2, 3, 4, 5, 8, 10])))
            rows.append({"question": q, "final": normalize(gold), "cot": body,
                         "source": "rft_self",
                         "answer": build_completion(body, normalize(gold)).replace(EOT, "").strip(),
                         "prompt": build_prompt(q, block),
                         "completion": build_completion(body, normalize(gold))})

    rng.shuffle(rows)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats = {"questions": len(items), "solved_at_least_once": n_solved,
             "solve_rate": round(n_solved / len(items), 4),
             "mean_correct_per_question": round(sum(per_q_correct) / len(items), 3),
             "rows": len(rows), "out": a.out}
    json.dump(stats, open(a.stats_out, "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
