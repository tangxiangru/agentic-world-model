#!/usr/bin/env python3
"""One vLLM load, three probes on the held-out train-split probe set:

  1. greedy with the grader's 10-shot prefix   (matches the graded condition)
  2. greedy zero-shot                          (matches 90% of the training rows)
  3. pass@k at temperature 1.0, 10-shot        (headroom available to a rejection-
                                                sampling round)
"""
from __future__ import annotations

import argparse
import json

import prompt_spec as ps
from rft_sample import last_number, norm


def score(rows, outs, idx=0):
    correct = capped = no_marker = 0
    for r, o in zip(rows, outs):
        c = o.outputs[idx]
        if c.finish_reason == "length":
            capped += 1
        if "ANSWER:" not in c.text:
            no_marker += 1
        if last_number(c.text) == norm(r["gold"]):
            correct += 1
    return {"accuracy": correct / len(rows), "hit_cap": capped, "no_marker": no_marker}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", default="data/probe300.jsonl")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-passk", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.probe)]
    sysmsg = ps.fewshot_system_message()
    p_fs = [ps.render_prompt(r["question"], sysmsg) for r in rows]
    p_zs = [ps.render_prompt(r["question"], None) for r in rows]

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, enable_prefix_caching=True)
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop_token_ids=[1, 106])

    res = {"model": args.model, "n": len(rows)}
    res["greedy_10shot"] = score(rows, llm.generate(p_fs, greedy))
    res["greedy_zeroshot"] = score(rows, llm.generate(p_zs, greedy))

    if not args.no_passk:
        sp = SamplingParams(n=args.k, temperature=1.0, top_p=0.95,
                            max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=0)
        outs = llm.generate(p_fs, sp)
        any_ok = sum(
            any(last_number(c.text) == norm(r["gold"]) for c in o.outputs)
            for r, o in zip(rows, outs)
        )
        per_sample = sum(
            sum(last_number(c.text) == norm(r["gold"]) for c in o.outputs) / args.k
            for r, o in zip(rows, outs)
        ) / len(rows)
        res[f"pass_at_{args.k}_t1"] = any_ok / len(rows)
        res["mean_sample_acc_t1"] = per_sample

    # failures under the graded condition, for the next card's watch set
    fs_outs = llm.generate(p_fs, greedy)
    fails = [{"id": r["id"], "gold": r["gold"], "question": r["question"],
              "output": o.outputs[0].text[-700:]}
             for r, o in zip(rows, fs_outs)
             if last_number(o.outputs[0].text) != norm(r["gold"])]
    res["n_failures"] = len(fails)
    print(json.dumps(res, indent=2))
    with open(args.out, "w") as f:
        json.dump({"summary": res, "failures": fails}, f, indent=2)


if __name__ == "__main__":
    main()
