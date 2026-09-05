"""Run several decode/prompt configurations against the probe set in one vLLM load."""
import argparse
import json
import os
import re
from collections import Counter

from transformers import AutoTokenizer

from probe_eval import MATH_PROMPT_TEMPLATE, SNAP, TEMPLATE, last_number, norm_num

CONFIGS = [
    {"name": "zeroshot_greedy", "fewshot": False, "temperature": 0.0},
    {"name": "fewshot_greedy", "fewshot": True, "temperature": 0.0},
    {"name": "fewshot_sampled_evaldefault", "fewshot": True, "temperature": 1.0,
     "top_p": 0.95, "top_k": 64},
    {"name": "zeroshot_sampled_evaldefault", "fewshot": False, "temperature": 1.0,
     "top_p": 0.95, "top_k": 64},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", default="/home/ben/task/data/probe_gsm8ktrain300.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--configs", default="")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(SNAP)
    template = open(TEMPLATE).read()
    items = [json.loads(l) for l in open(args.probe)][: args.n]
    sysmsg = open("/home/ben/task/analysis/fewshot_system_message.txt").read()

    def build(fewshot):
        out = []
        for it in items:
            user = MATH_PROMPT_TEMPLATE.format(prompt=it["question"])
            msgs = ([{"role": "system", "content": sysmsg}] if fewshot else []) + \
                   [{"role": "user", "content": user}]
            out.append(tok.apply_chat_template(msgs, chat_template=template,
                                               tokenize=False, add_generation_prompt=True))
        return out

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=4096,
              dtype="bfloat16", disable_log_stats=True)

    wanted = set(args.configs.split(",")) if args.configs else None
    summaries = {}
    details = {}
    for cfg in CONFIGS:
        if wanted and cfg["name"] not in wanted:
            continue
        sp = SamplingParams(
            temperature=cfg["temperature"],
            top_p=cfg.get("top_p", 1.0),
            top_k=cfg.get("top_k", -1),
            max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=0,
        )
        outs = llm.generate(build(cfg["fewshot"]), sp)
        n_ok = n_fmt = n_trunc = 0
        recs = []
        for it, o in zip(items, outs):
            t = o.outputs[0].text
            p = last_number(t)
            pred = norm_num(p) if p is not None else None
            ok = pred is not None and pred == norm_num(it["gold"])
            n_ok += ok
            n_fmt += bool(re.search(r"^ANSWER: ", t, re.M))
            n_trunc += o.outputs[0].finish_reason == "length"
            recs.append({"id": it["id"], "gold": it["gold"], "pred": pred,
                         "correct": bool(ok), "output": t})
        summaries[cfg["name"]] = {"accuracy": n_ok / len(items),
                                  "format_compliance": n_fmt / len(items),
                                  "truncated": n_trunc / len(items), "n": len(items)}
        details[cfg["name"]] = recs
        print(cfg["name"], json.dumps(summaries[cfg["name"]]), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"model": args.model, "summary": summaries, "details": details},
              open(args.out, "w"))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
