#!/usr/bin/env python3
"""Rejection-sampling fine-tuning data: sample k solutions per training problem
from a checkpoint, keep only those whose 'ANSWER: <n>' matches the reference
answer, dedup, and cap per problem.

Problems come from the GSM8K *train* split and from OpenMathInstruct-2's
gsm8k-derived problems (also train-only). The test split is never touched.
"""
import argparse, collections, json, os, random, re

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")


def norm_int(s):
    s = str(s).strip().replace(",", "").replace("$", "").rstrip(".")
    if s.endswith(".0"):
        s = s[:-2]
    return s if re.fullmatch(r"-?\d+", s) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n-aug", type=int, default=12500,
                    help="how many distinct augmented_gsm8k problems to add")
    ap.add_argument("--max-keep-per-problem", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rng = random.Random(a.seed)
    template = open("/home/ben/task/templates/gemma3.jinja").read()
    tok = AutoTokenizer.from_pretrained(a.model)

    probs = []  # (question, gold_int, source)
    g = load_dataset("openai/gsm8k", "main", split="train")
    for r in g:
        gold = norm_int(r["answer"].rpartition("####")[2])
        if gold is not None:
            probs.append((r["question"], gold, "gsm8k_train"))

    if a.n_aug > 0:
        d = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
        seen = {}
        for r in d:
            if r["problem_source"] != "augmented_gsm8k":
                continue
            if r["problem"] in seen:
                continue
            gold = norm_int(r["expected_answer"])
            if gold is None:
                continue
            seen[r["problem"]] = gold
        items = list(seen.items())
        rng.shuffle(items)
        for q, gold in items[: a.n_aug]:
            probs.append((q, gold, "omi2_augmented_gsm8k"))

    print(f"[rft] {len(probs)} problems x k={a.k} = {len(probs)*a.k} generations")

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q.strip())}],
            chat_template=template, add_generation_prompt=True, tokenize=False)
        for q, _, _ in probs
    ]

    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=2048,
              dtype="bfloat16", enforce_eager=False, seed=a.seed)
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p,
                        max_tokens=a.max_tokens, stop_token_ids=[1, 106], seed=a.seed)
    outs = llm.generate(prompts, sp)

    rows, kept_per, n_corr, n_tot = [], collections.Counter(), 0, 0
    for (q, gold, src), o in zip(probs, outs):
        seen_text = set()
        for c in o.outputs:
            n_tot += 1
            t = c.text.strip()
            m = ANS_RE.search(t)
            if not m:
                continue
            v = norm_int(m.group(1))
            if v is None or v != gold:
                continue
            # exactly one answer marker, and it must be the last line
            if t.count("ANSWER:") != 1:
                continue
            if not t.splitlines()[-1].strip().startswith("ANSWER:"):
                continue
            n_corr += 1
            if t in seen_text or kept_per[q] >= a.max_keep_per_problem:
                continue
            seen_text.add(t)
            kept_per[q] += 1
            rows.append({
                "prompt": MATH_PROMPT_TEMPLATE.format(prompt=q.strip()),
                "completion": t + "<end_of_turn>",
                "answer": gold,
                "source": "rft_" + src,
                "text": q.strip() + "\n" + t,
            })

    rng.shuffle(rows)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[rft] generations={n_tot} correct={n_corr} ({n_corr/max(1,n_tot):.3f}) "
          f"kept={len(rows)} problems_covered={len(kept_per)}/{len(probs)}")
    print(collections.Counter(r["source"] for r in rows))


if __name__ == "__main__":
    main()
