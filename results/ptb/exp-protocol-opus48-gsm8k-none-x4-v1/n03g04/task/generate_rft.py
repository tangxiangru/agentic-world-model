#!/usr/bin/env python3
"""Rejection sampling: sample K CoT solutions per GSM8K train question from the
current SFT model, keep those whose final number matches the gold answer.
Output rft_data.jsonl with {question, reasoning_answer}."""
import json, re, os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL = "sft_out3"
TPL = open("templates/gemma3.jinja").read()
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def last_number(t):
    t = re.sub(r"[,$]", "", t)
    for w in reversed(t.split()):
        w2 = w.rstrip(".")
        if re.fullmatch(r"-?\d+(\.\d+)?", w2):
            return w2
    return None


def norm(s):
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return s


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    prompts = []
    golds = []
    questions = []
    for r in gsm:
        q = r["question"].strip()
        gold = r["answer"].split("####")[1].strip().replace(",", "").replace("$", "")
        msgs = [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    chat_template=TPL)
        prompts.append(p)
        golds.append(gold)
        questions.append(q)

    llm = LLM(model=MODEL, gpu_memory_utilization=0.85, max_model_len=2048,
              dtype="bfloat16", enforce_eager=False)
    sp = SamplingParams(n=6, temperature=0.9, top_p=0.95, max_tokens=512,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    kept = {}
    total_correct = 0
    for i, o in enumerate(outs):
        gold_n = norm(golds[i])
        seen = set()
        for comp in o.outputs:
            text = comp.text.strip()
            ln = last_number(text)
            if ln is None:
                continue
            if norm(ln) != gold_n:
                continue
            total_correct += 1
            # clean: cut anything after the ANSWER line
            m = re.search(r"ANSWER:\s*[^\n]*", text)
            if m:
                text = text[:m.end()]
            # dedup by normalized reasoning text
            key = re.sub(r"\s+", " ", text)[:200]
            if key in seen:
                continue
            seen.add(key)
            kept.setdefault(i, []).append(text)

    with open("rft_data.jsonl", "w") as f:
        nex = 0
        for i, sols in kept.items():
            # keep up to 3 diverse correct solutions per question
            for s in sols[:3]:
                f.write(json.dumps({"question": questions[i],
                                    "reasoning_answer": s}) + "\n")
                nex += 1
    print(f"questions solved: {len(kept)}/{len(questions)}  correct samples: {total_correct}  written: {nex}")


if __name__ == "__main__":
    main()
