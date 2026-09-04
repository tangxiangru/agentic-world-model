"""Fast offline eval on the PRIVATE dev set (held-out GSM8K *train* items).

Not a substitute for evaluate.py -- the official protocol stays evaluate.py --limit 150.
This exists so iteration and failure analysis never touch the benchmark test split
(protocol rule 7), and so a 300-item read costs ~2 min instead of ~15.

Renders prompts through templates/gemma3.jinja and scores the way inspect's
match(numeric=True, location='end') does: the last numeric token of the completion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402
from rft_sample import extract  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

SNAPSHOT = os.environ["PTB_BASE_MODEL_SNAPSHOT"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", default="data/dev300.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fails", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--fewshot", type=int, default=0,
                    help="prepend k GSM8K-train shots the way the grader's system message does")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    tok.chat_template = fmt.load_template()

    dev = [json.loads(l) for l in open(args.dev)]
    if args.limit:
        dev = dev[: args.limit]

    shots = None
    if args.fewshot:
        from datasets import load_dataset
        g = load_dataset("openai/gsm8k", "main", split="train")
        shots = []
        for r in g.select(range(args.fewshot)):
            a = fmt.normalize_answer(r["answer"].split("####")[-1])
            body = fmt.strip_gsm8k_calc(r["answer"].split("####")[0]).strip()
            shots.append((r["question"], body, a))

    prompts = []
    for d in dev:
        msgs = []
        if shots:
            msgs.append({"role": "system", "content": fmt.fewshot_block(shots)})
        msgs.append({"role": "user", "content": fmt.user_prompt(d["question"])})
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, dtype="bfloat16")
    sp = SamplingParams(temperature=args.temp, top_p=args.top_p, top_k=args.top_k,
                        max_tokens=args.max_tokens, stop=[fmt.STOP_TOKEN])
    outs = llm.generate(prompts, sp)

    n_ok = n_fmt = n_cap = 0
    fails = []
    for d, o in zip(dev, outs):
        c = o.outputs[0]
        text = c.text.strip()
        got = extract(text)
        ok = got == d["gold"]
        n_ok += ok
        lines = [x for x in text.split("\n") if x.strip()]
        good_fmt = bool(lines) and lines[-1].strip().startswith("ANSWER:")
        n_fmt += good_fmt
        capped = c.finish_reason == "length"
        n_cap += capped
        if not ok:
            fails.append({"id": d["id"], "question": d["question"], "gold": d["gold"],
                          "extracted": got, "capped": capped, "format_ok": good_fmt,
                          "completion": text[-1500:]})

    n = len(dev)
    res = {"model": args.model, "dev": args.dev, "n": n, "temp": args.temp,
           "top_p": args.top_p, "top_k": args.top_k,
           "fewshot": args.fewshot, "accuracy": n_ok / n, "format_ok": n_fmt / n,
           "cap_hit": n_cap / n}
    print(json.dumps(res, indent=2))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    if args.fails:
        with open(args.fails, "w") as f:
            for r in fails:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
