#!/usr/bin/env python3
"""Fast offline dev eval on data/dev_holdout.jsonl (gsm8k TRAIN items held out of
training).  Reproduces the grader exactly: the same 10-shot system prefix, the
same MATH_PROMPT_TEMPLATE, templates/gemma3.jinja, greedy decoding, and
inspect's match(numeric=True, location="end") rule (last numeric token).

This is a diagnostic, not the benchmark: the benchmark is evaluate.py --limit 150.
"""
import argparse
import json
import re

from datasets import load_from_disk
from transformers import AutoTokenizer

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC = re.compile(r"<<[^>]*>>")


def normalize_number(s):
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f == int(f) else f"{f:.5f}".rstrip("0")


def last_number_normalized(text):
    for w in reversed(re.split(r"\s+", text.strip())):
        w2 = w.replace(",", "").replace("$", "").rstrip(".:;)%")
        if w2.replace(".", "").replace("-", "").isnumeric():
            return normalize_number(w2)
    return None


def fewshot_system():
    gsm = load_from_disk("data/gsm8k_raw")["train"].shuffle(seed=42).select(range(10))
    out = []
    for r in gsm:
        body, final = r["answer"].split("####")
        # NB: inspect's sample_to_fewshot keeps the <<...>> calculator annotations;
        # stripping them here would make this proxy's prompt differ from the graded one.
        out.append(f"{r['question']}\n\nReasoning:\n{body.strip()}\n\nANSWER: {final.strip()}")
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", default="data/dev_holdout.jsonl")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-fewshot", action="store_true",
                    help="drop the 10-shot system prefix (probe: is the prefix helping or hurting?)")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.dev)][: a.limit]
    tok = AutoTokenizer.from_pretrained(a.model)
    tmpl = open("templates/gemma3.jinja").read()
    sysmsg = None if a.no_fewshot else fewshot_system()
    prompts = []
    for r in rows:
        msgs = ([] if sysmsg is None else [{"role": "system", "content": sysmsg}]) + \
               [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=r["question"])}]
        prompts.append(tok.apply_chat_template(msgs, chat_template=tmpl, tokenize=False,
                                               add_generation_prompt=True))

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_memory_utilization,
              max_model_len=8192, dtype="bfloat16", enforce_eager=False)
    sp = SamplingParams(temperature=a.temperature, max_tokens=a.max_tokens, n=a.n,
                        top_p=1.0 if a.temperature == 0 else 0.95,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    n_ok = n_trunc = n_noans = 0
    recs = []
    for r, o in zip(rows, outs):
        texts = [c.text for c in o.outputs]
        got = last_number_normalized(texts[0])
        gold = normalize_number(r["gold"].replace(",", ""))
        ok = got is not None and got == gold
        n_ok += ok
        n_trunc += o.outputs[0].finish_reason == "length"
        n_noans += "ANSWER:" not in texts[0]
        recs.append({"question": r["question"], "gold": gold, "got": got, "ok": bool(ok),
                     "finish": o.outputs[0].finish_reason, "text": texts[0]})
    stats = {"model": a.model, "n": len(rows), "accuracy": round(n_ok / len(rows), 4),
             "truncated": n_trunc, "no_answer_line": n_noans,
             "mean_out_tokens": round(sum(len(o.outputs[0].token_ids) for o in outs) / len(outs), 1)}
    print(json.dumps(stats, indent=2))
    if a.out:
        json.dump({"stats": stats, "records": recs}, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
