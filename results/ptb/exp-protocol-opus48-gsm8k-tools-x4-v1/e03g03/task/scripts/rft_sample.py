#!/usr/bin/env python3
"""RFT/STaR sampling: draw solutions from exp-02 for GSM8K-train questions,
keep those whose last number equals the gold answer.

vLLM pitfalls handled:
- prompts already contain <bos>: pass prompt_token_ids (no second special token)
- explicit stop_token_ids = [1, 106] (<eos>, <end_of_turn>)
- raw draws saved BEFORE filtering
"""
import argparse, json, os, re, sys
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def norm(x):
    return x.strip().rstrip(".").replace(",", "").replace("$", "").strip()


def last_number(t):
    for w in re.split(r"\s+", t.strip())[::-1]:
        w2 = re.sub(r"[^0-9\.\-]", "", w)
        if re.search(r"\d", w2):
            return w2.rstrip(".")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/ben/task/ckpts/exp-02/final")
    ap.add_argument("--data", default="/home/ben/task/data/gsm8k_train.jsonl")
    ap.add_argument("--raw_out", default="/home/ben/task/data/rft_raw.jsonl")
    ap.add_argument("--out", default="/home/ben/task/data/rft_correct.jsonl")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit > 0:
        rows = rows[:args.limit]
    tok = AutoTokenizer.from_pretrained(SNAP)
    prompt_ids = [tok.encode(r["prompt"], add_special_tokens=False) for r in rows]
    assert all(p[0] == 2 for p in prompt_ids), "prompt must start with single <bos>=2"

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=1280, enforce_eager=False)
    params = SamplingParams(n=args.n, temperature=args.temp, top_p=0.95,
                            max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=0)
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids]
    outs = llm.generate(prompts, sampling_params=params)

    # preserve raw draws first
    kept, n_correct_samples, n_q_solved = [], 0, 0
    with open(args.raw_out, "w") as rf:
        for r, o in zip(rows, outs):
            gold = norm(r["answer"])
            texts = [c.text for c in o.outputs]
            rf.write(json.dumps({"answer": gold, "samples": texts}) + "\n")
            seen = set(); good = []
            for t in texts:
                ln = last_number(t)
                ok = False
                if ln is not None:
                    try:
                        ok = abs(float(ln) - float(gold)) < 1e-6
                    except Exception:
                        ok = (ln == gold)
                if ok and t.count("ANSWER:") == 1:
                    comp = t.rstrip()
                    if not comp.endswith("<end_of_turn>"):
                        comp = comp + "<end_of_turn>"
                    key = comp.strip()
                    if key not in seen:
                        seen.add(key); good.append(comp)
                        n_correct_samples += 1
            good = good[:args.cap]
            if good:
                n_q_solved += 1
            for comp in good:
                kept.append({"prompt": r["prompt"], "completion": comp, "answer": gold})

    with open(args.out, "w") as f:
        for k in kept:
            f.write(json.dumps(k) + "\n")
    print(f"questions={len(rows)} solved>=1={n_q_solved} ({n_q_solved/len(rows):.3f}) "
          f"kept_solutions={len(kept)}", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)  # force clean exit; avoid vLLM engine shutdown hang


if __name__ == "__main__":
    main()
