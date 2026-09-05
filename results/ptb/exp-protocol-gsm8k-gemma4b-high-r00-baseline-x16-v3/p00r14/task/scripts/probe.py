"""Score a checkpoint on a held-out GSM8K *train* slice, offline with vLLM.

This is the dev harness the cards use for failure_examples and watch sets: the
benchmark test copy may not appear in either, so the probe set is 250 GSM8K
train items that build_sft_data.py excluded from training. Prompts are rendered
with the grader's own template and 10-shot system message, and graded with
inspect's own match_str, so a probe number means the same thing an evaluate.py
number does - only on different items.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(TASK_DIR, "data",
                                                   "dev_heldout250.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--fewshot", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from inspect_ai.scorer._match import match_str
    from fmt import render_prompt

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    system = None
    if args.fewshot:
        with open(os.path.join(TASK_DIR, "data", "eval_fewshot_system.txt")) as f:
            system = f.read()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    prompts = [render_prompt(tokenizer, r["question"], system) for r in rows]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=4096, enforce_eager=False)
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    sp = SamplingParams(temperature=args.temperature,
                        top_p=1.0 if args.temperature == 0 else 0.95,
                        max_tokens=args.max_tokens, n=args.n,
                        stop=["<end_of_turn>"], stop_token_ids=[eot])
    outs = llm.generate(prompts, sp)

    n_correct = 0
    recs = []
    for r, o in zip(rows, outs):
        texts = [c.text for c in o.outputs]
        oks = [match_str(value=t, target=r["gold"], location="end", numeric=True)[1]
               for t in texts]
        ok = oks[0]
        n_correct += int(ok)
        recs.append({"id": r["id"], "question": r["question"], "gold": r["gold"],
                     "outputs": texts, "correct": oks})
    acc = n_correct / len(rows)
    pass_at_n = sum(any(r["correct"]) for r in recs) / len(rows)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "n": len(rows), "accuracy": acc,
                   "pass_at_n": pass_at_n, "temperature": args.temperature,
                   "samples": recs}, f)
    print(json.dumps({"model": args.model, "n": len(rows), "accuracy": acc,
                      "pass_at_n": pass_at_n}), flush=True)


if __name__ == "__main__":
    main()
