"""Fast offline probe eval with vLLM on held-out GSM8K TRAIN problems.

Uses the grader's prompt template and the grader's answer rule (last numeric
whitespace token of the completion), so the number tracks the official protocol
without spending its minutes. Optionally prepends the same 10-shot system
message the official eval uses, to measure the zero-shot/few-shot gap.
"""
import argparse
import json
import os
import re

from transformers import AutoTokenizer

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"


def last_number(text):
    words = re.split(r"\s+", text.strip())
    for w in reversed(words):
        c = w.replace(",", "").replace("$", "").rstrip(".").rstrip("*")
        if c.replace(".", "").replace("-", "").isnumeric():
            return c.lstrip("-") if c.count("-") > 1 else c
    return None


def norm_num(s):
    try:
        return format(float(s), ".5g")
    except Exception:
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", default="/home/ben/task/data/probe_gsm8ktrain300.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=1, help="samples per item (majority vote if >1)")
    ap.add_argument("--fewshot", action="store_true", help="prepend the official 10-shot system message")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(SNAP)
    template = open(TEMPLATE).read()
    items = [json.loads(l) for l in open(args.probe)][: args.n]

    sysmsg = None
    if args.fewshot:
        sysmsg = open("/home/ben/task/analysis/fewshot_system_message.txt").read()

    prompts = []
    for it in items:
        user = MATH_PROMPT_TEMPLATE.format(prompt=it["question"])
        msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + \
               [{"role": "user", "content": user}]
        prompts.append(tok.apply_chat_template(
            msgs, chat_template=template, tokenize=False, add_generation_prompt=True))

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=4096,
              dtype="bfloat16", enforce_eager=False, disable_log_stats=True)
    sp = SamplingParams(temperature=args.temperature, top_p=1.0 if args.temperature == 0 else 0.95,
                        max_tokens=args.max_tokens, n=args.k, stop_token_ids=[1, 106], seed=0)
    outs = llm.generate(prompts, sp)

    n_ok = n_fmt = n_trunc = 0
    recs = []
    for it, o in zip(items, outs):
        texts = [c.text for c in o.outputs]
        preds = [last_number(t) for t in texts]
        if args.k > 1:
            from collections import Counter
            cnt = Counter(norm_num(p) for p in preds if p is not None)
            pred = cnt.most_common(1)[0][0] if cnt else None
        else:
            pred = norm_num(preds[0]) if preds[0] is not None else None
        ok = pred is not None and pred == norm_num(it["gold"])
        n_ok += ok
        n_fmt += bool(re.search(r"^ANSWER: ", texts[0], re.M))
        n_trunc += (o.outputs[0].finish_reason == "length")
        recs.append({"id": it["id"], "gold": it["gold"], "pred": pred, "correct": bool(ok),
                     "output": texts[0]})

    res = {"model": args.model, "n": len(items), "k": args.k, "temperature": args.temperature,
           "fewshot": bool(args.fewshot), "accuracy": n_ok / len(items),
           "format_compliance": n_fmt / len(items), "truncated": n_trunc / len(items)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": res, "records": recs}, f, indent=1)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
