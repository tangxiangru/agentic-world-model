"""Cheap off-benchmark probe: score a checkpoint on data/probe200.jsonl
(held-out GSM8K *train* items) using the grader's exact prompt rendering,
greedy decoding, and the grader's answer convention (last number wins).

This never touches the benchmark test set; it is the watch set the cards refer to.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_number(text: str) -> str | None:
    m = NUM_RE.findall(text.replace("$", "").replace("*", ""))
    if not m:
        return None
    v = m[-1].replace(",", "")
    if v.endswith("."):
        v = v[:-1]
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", default="/home/ben/task/data/probe200.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fewshot", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    items = [json.loads(l) for l in open(args.probe)]
    prompts = [fmt.render_prompt(it["question"], fewshot=bool(args.fewshot)) for it in items]
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=4096,
        dtype="bfloat16",
        seed=0,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    recs, n_ok, n_fmt = [], 0, 0
    for it, o in zip(items, outs):
        t = o.outputs[0].text.strip()
        pred = final_number(t)
        ok = pred is not None and pred.rstrip(".0") == it["gold"].rstrip(".0") or pred == it["gold"]
        n_ok += bool(ok)
        n_fmt += bool(re.search(r"ANSWER:\s*-?[\d,\.]+\s*$", t))
        recs.append({"id": it["id"], "gold": it["gold"], "pred": pred, "correct": bool(ok), "output": t})
    res = {
        "model": args.model,
        "n": len(items),
        "accuracy": n_ok / len(items),
        "ends_with_answer_line": n_fmt / len(items),
        "fewshot_prompt": bool(args.fewshot),
        "samples": recs,
    }
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[probe] n={len(items)} accuracy={res['accuracy']:.3f} format={res['ends_with_answer_line']:.3f}")


if __name__ == "__main__":
    main()
