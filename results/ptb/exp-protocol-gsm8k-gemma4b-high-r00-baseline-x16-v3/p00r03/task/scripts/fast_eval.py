"""Fast dev evaluation with vLLM, rendering and grading exactly like evaluate.py.

Not the official protocol -- used for diagnostics and model selection on a
held-out slice of the GSM8K *train* split.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    GEMMA_TEMPLATE,
    fewshot_system_message,
    grade,
    read_jsonl,
    user_prompt,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dev500.jsonl"))
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--fewshot", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    rows = read_jsonl(args.data)[: args.limit]

    with open(GEMMA_TEMPLATE) as f:
        chat_template = f.read()

    sys_msg = None
    if args.fewshot:
        sys_msg, _ = fewshot_system_message(n=args.fewshot)

    convs = []
    for r in rows:
        msgs = []
        if sys_msg:
            msgs.append({"role": "system", "content": sys_msg})
        msgs.append({"role": "user", "content": user_prompt(r["question"])})
        convs.append(msgs)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        enforce_eager=False,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=1.0 if args.temperature == 0 else 0.95,
        max_tokens=args.max_tokens,
        n=args.n,
        seed=0 if args.temperature == 0 else None,
    )
    outs = llm.chat(convs, sp, chat_template=chat_template, add_generation_prompt=True)

    recs, correct = [], 0
    for r, o in zip(rows, outs):
        texts = [c.text for c in o.outputs]
        oks = [grade(t, r["gold"]) for t in texts]
        ok = oks[0]
        correct += int(ok)
        recs.append(
            {
                "id": r["id"],
                "question": r["question"],
                "gold": r["gold"],
                "outputs": texts,
                "correct": oks,
                "any_correct": any(oks),
            }
        )

    acc = correct / len(rows)
    any_acc = sum(r["any_correct"] for r in recs) / len(rows)
    summary = {
        "model": args.model_path,
        "n": len(rows),
        "fewshot": args.fewshot,
        "temperature": args.temperature,
        "samples_per_item": args.n,
        "accuracy": acc,
        "pass_at_n": any_acc,
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "records": recs}, f)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
