"""Fast offline probe eval on the held-out GSM8K-*train* problems.

Not a substitute for evaluate.py - the card's protocol number always comes from
evaluate.py.  This is the cheap diagnostic used for failure analysis, and its
prompt rendering is the same fmt.render_prompt the trainer used.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")


def extract(text):
    """Mirror inspect's match(location='end', numeric=True)."""
    v = text.strip().replace(",", "").replace("$", "")
    words = re.split(r"\s+", v)
    words.reverse()
    for w in words:
        w = w.strip().rstrip(".").rstrip("%")
        if w.replace(".", "").replace("-", "").isnumeric():
            try:
                return format(float(w), ".5g")
            except ValueError:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", default="/home/ben/task/data/probe200.jsonl")
    ap.add_argument("--shots", default="/home/ben/task/data/fewshot10.json")
    ap.add_argument("--n-shot", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.probe)]
    shots = json.load(open(args.shots))[: args.n_shot]
    system = "\n\n".join(shots) if shots else None
    prompts = [fmt.render_prompt(r["question"], system) for r in rows]

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=8192, dtype="bfloat16", enforce_eager=False)
    sp = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens,
                        n=args.n, stop_token_ids=[1, 106], seed=0)
    outs = llm.generate(prompts, sp)

    n_ok = 0
    recs = []
    for r, o in zip(rows, outs):
        texts = [c.text for c in o.outputs]
        preds = [extract(t) for t in texts]
        gold = format(float(r["gold"].replace(",", "")), ".5g")
        correct = any(p == gold for p in preds) if args.n > 1 else preds[0] == gold
        n_ok += bool(correct)
        recs.append({"id": r["id"], "question": r["question"], "gold": r["gold"],
                     "pred": preds[0], "correct": bool(correct), "output": texts[0]})
    acc = n_ok / len(rows)
    json.dump({"model": args.model, "n": len(rows), "n_shot": args.n_shot,
               "temperature": args.temperature, "samples_per_item": args.n,
               "accuracy": acc, "records": recs},
              open(args.out, "w"), indent=1)
    print(f"PROBE accuracy {acc:.4f} ({n_ok}/{len(rows)}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
