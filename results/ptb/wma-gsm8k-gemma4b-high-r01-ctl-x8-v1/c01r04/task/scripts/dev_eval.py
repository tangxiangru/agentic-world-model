#!/usr/bin/env python3
"""Internal dev probe: score a checkpoint on held-out GSM8K *train* problems.

Uses the same rendered prompt (10-shot, templates/gemma3.jinja) and the same
scoring rule as the grader (inspect_ai match(numeric=True, location='end')), so
numbers move with the official metric -- but on items the benchmark never sees.
This is a diagnostic, not the protocol metric; the protocol metric is evaluate.py.
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_format import fewshot_prefix, render_prompt
from inspect_ai._util.text import strip_numeric_punctuation


def scorer_reads(text: str) -> str:
    v = strip_numeric_punctuation(text.strip().casefold())
    words = re.split(r"\s+", v)
    words.reverse()
    return next((w for w in words if w.replace(".", "").isnumeric()), words[0] if words else "")


def norm(x: str) -> str:
    x = strip_numeric_punctuation(str(x).strip().casefold())
    try:
        return format(float(x), ".5g")
    except ValueError:
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", default="/home/ben/task/data/dev_gsm8ktrain.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--fewshot", type=int, default=10)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.dev)]
    if a.limit:
        rows = rows[: a.limit]
    prefix = fewshot_prefix(a.fewshot) if a.fewshot else None
    prompts = ["<bos>" + render_prompt(r["question"], prefix) for r in rows]

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_frac, max_model_len=4096,
              dtype="bfloat16", enforce_eager=False, disable_log_stats=True)
    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                        n=a.n, max_tokens=a.max_tokens, seed=0,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    res, ncorrect = [], 0
    for r, o in zip(rows, outs):
        texts = [c.text for c in o.outputs]
        preds = [scorer_reads(t) for t in texts]
        gold = norm(r["gold"])
        maj = max(set(preds), key=preds.count) if a.n > 1 else preds[0]
        ok = norm(maj) == gold
        ncorrect += ok
        res.append({"id": r["id"], "gold": r["gold"], "pred": maj, "correct": bool(ok),
                    "n_stop": sum(1 for c in o.outputs if c.finish_reason == "stop"),
                    "text": texts[0]})
    acc = ncorrect / len(rows)
    summary = {"model": a.model, "n": len(rows), "accuracy": acc,
               "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k,
               "n_samples": a.n, "fewshot": a.fewshot,
               "stop_rate": sum(r["n_stop"] for r in res) / (len(rows) * a.n)}
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "items": res}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
