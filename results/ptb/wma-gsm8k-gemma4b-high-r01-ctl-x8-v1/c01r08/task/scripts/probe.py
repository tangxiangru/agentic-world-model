"""Fast local probe on the held-out GSM8K-train items.

Renders prompts with scripts/grader_format.py (byte-identical to
templates/gemma3.jinja) and scores with the replica of
inspect_ai.scorer.match(numeric=True, location="end"), so a number here means
the same thing a number from evaluate.py means -- on a different item set.

Sweeps decode settings and the presence of the grader's 10-shot system prefix
in one vLLM process, which is what makes it cheap enough to run between runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grader_format as gf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", default="data/dev_train300.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--conditions", default="greedy_few,sample_few,greedy_zero,sample_zero")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.dev)][:args.n]
    sysmsg = gf.build_fewshot_system()

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=4096,
              dtype="bfloat16", seed=0)

    results = {}
    for cond in args.conditions.split(","):
        decode, shots = cond.split("_")
        system = sysmsg if shots == "few" else None
        prompts = [gf.render_prompt(it["question"], system) for it in items]
        if decode == "greedy":
            sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        else:
            sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=64,
                                max_tokens=args.max_tokens, seed=0)
        outs = llm.generate(prompts, sp)
        ok = nostop = 0
        rows = []
        for it, o in zip(items, outs):
            t = o.outputs[0].text
            stop = o.outputs[0].finish_reason
            good = gf.score_completion(t.split(gf.STOP_TOKEN)[0], it["gold"])
            ok += good
            nostop += (stop != "stop")
            rows.append({"id": it["id"], "ok": bool(good), "finish": stop, "tail": t[-160:]})
        results[cond] = {"accuracy": ok / len(items), "n": len(items),
                         "not_stopped": nostop, "rows": rows}
        print(f"{cond:14s} acc={ok/len(items):.4f}  not_stopped={nostop}", flush=True)

    json.dump({"model": args.model, "dev": args.dev,
               "summary": {k: {"accuracy": v["accuracy"], "n": v["n"],
                               "not_stopped": v["not_stopped"]} for k, v in results.items()},
               "detail": results}, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
