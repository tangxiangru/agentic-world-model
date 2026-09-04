"""Fast offline dev scorer on held-out GSM8K TRAIN questions.

Reproduces the grader byte-for-byte where it matters: the same 10-shot system
message inspect_evals builds (train split, shuffle, seed 42, limit 10), the same
MATH_PROMPT_TEMPLATE, the same templates/gemma3.jinja render, and the same
match(location="end", numeric=True) rule -- the LAST number in the completion
must equal the gold answer.  Only the item pool differs: data/dev300_gsm8ktrain
instead of the benchmark test split, so no test item is ever read.
"""
import argparse, json, os, re, sys

NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")
PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def norm(s):
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    return int(f) if f == int(f) else round(f, 6)


def last_number(text):
    ms = NUM_RE.findall(text)
    return norm(ms[-1]) if ms else None


def fewshot_system():
    import re as _re
    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42)
    parts = []
    for r in list(gsm)[:10]:
        reasoning, _, tgt = r["answer"].rpartition("####")
        parts.append(f"{r['question']}\n\nReasoning:\n{reasoning.strip()}\n\n"
                     f"ANSWER: {tgt.strip()}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default="data/dev300_gsm8ktrain.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--fewshot", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.model)
    tpl = open("/home/ben/task/templates/gemma3.jinja").read()
    items = [json.loads(l) for l in open(a.items)][: a.n]
    sysmsg = fewshot_system() if a.fewshot else None

    prompts = []
    for it in items:
        user = PROMPT_TEMPLATE.format(prompt=it["question"])
        msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + \
               [{"role": "user", "content": user}]
        prompts.append(tok.apply_chat_template(msgs, chat_template=tpl,
                                               tokenize=False,
                                               add_generation_prompt=True))

    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, dtype="bfloat16",
              max_model_len=8192, enforce_eager=False)
    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                        max_tokens=a.max_tokens, seed=0)
    outs = llm.generate(prompts, sp)

    correct = trunc = 0
    recs = []
    for it, o in zip(items, outs):
        txt = o.outputs[0].text
        got = last_number(txt)
        ok = got is not None and got == norm(it["gold"])
        correct += ok
        fin = o.outputs[0].finish_reason
        trunc += fin == "length"
        recs.append({"id": it["id"], "gold": it["gold"], "got": got, "ok": bool(ok),
                     "finish": fin, "ntok": len(o.outputs[0].token_ids),
                     "completion": txt})
    n = len(items)
    res = {"model": a.model, "n": n, "accuracy": correct / n, "truncated": trunc,
           "mean_tokens": sum(r["ntok"] for r in recs) / n,
           "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k,
           "fewshot": a.fewshot}
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump({**res, "samples": recs}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
