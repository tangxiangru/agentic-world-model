#!/usr/bin/env python3
"""Rejection sampling: generate k solutions per question with the SFT model,
keep the ones whose final ANSWER matches the reference."""
from __future__ import annotations
import argparse, glob, json, os, random, re
import pyarrow.parquet as pq
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def norm(s: str):
    s = s.strip().replace(",", "").replace("$", "").replace("*", "").rstrip(".")
    if not NUM_RE.match(s):
        return None
    f = float(s)
    return f"{f:.5g}"


def final_answer(text: str):
    m = re.findall(r"ANSWER:\s*([^\n]*)", text)
    if not m:
        return None
    return norm(m[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/rft_raw.jsonl")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-aug", type=int, default=6000)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    questions = []  # (question, normalized answer, tag)
    gsm = load_dataset("openai/gsm8k", "main")["train"]
    for r in gsm:
        t = r["answer"].split("####")[-1].strip().replace(",", "")
        n = norm(t)
        if n:
            questions.append((r["question"].strip(), n, "gsm_train"))

    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"))
    aug = {}
    for f in files:
        t = pq.read_table(f, columns=["problem", "expected_answer", "problem_source"]).to_pandas()
        t = t[t.problem_source == "augmented_gsm8k"]
        for p, a in zip(t.problem, t.expected_answer):
            n = norm(str(a))
            if n:
                aug.setdefault(str(p).strip(), n)
    aug_items = list(aug.items())
    rng.shuffle(aug_items)
    for p, a in aug_items[:args.n_aug]:
        questions.append((p, a, "aug"))
    print("questions:", len(questions))

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tpl = open("templates/gemma3.jinja").read()
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for q, a, tag in questions:
        msg = [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}]
        prompts.append(tok.apply_chat_template(msg, chat_template=tpl, tokenize=False,
                                               add_generation_prompt=True))
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util,
              max_model_len=2048, enforce_eager=False, dtype="bfloat16")
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=0.95, top_k=64,
                        max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate(prompts, sp)

    n_keep = 0
    with open(args.out, "w") as f:
        for (q, a, tag), o in zip(questions, outs):
            sols = []
            for c in o.outputs:
                if c.finish_reason != "stop":
                    continue
                txt = c.text.strip()
                if final_answer(txt) == a:
                    sols.append(txt)
            f.write(json.dumps({"question": q, "answer": a, "tag": tag,
                                "n_correct": len(sols), "n_total": len(o.outputs),
                                "solutions": sols}) + "\n")
            n_keep += len(sols)
    print("kept solutions:", n_keep)


if __name__ == "__main__":
    main()
