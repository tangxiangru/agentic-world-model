#!/usr/bin/env python3
"""Offline vLLM generation replicating the inspect_evals/gsm8k prompt exactly.

Used both for (a) fast local scoring on a GSM8K-train holdout and
(b) rejection-sampling data generation.
"""
import argparse, json, os, random, re, sys

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def eval_fewshot_system():
    """Exactly the 10-shot system message inspect builds (seed 42, shuffled)."""
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot
    fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                          sample_fields=record_to_sample, shuffle=True, seed=42,
                          limit=10)
    return "\n\n".join(sample_to_fewshot(s) for s in fewshots), \
        set(s.input.strip() for s in fewshots)


def score(completion: str, target: str) -> bool:
    from inspect_ai.scorer._common import match_str
    try:
        return match_str(completion, target, location="end", ignore_case=True,
                         numeric=True)[1]
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="jsonl with question/answer")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--fewshot", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    if args.limit:
        rows = rows[:args.limit]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ct = open("templates/gemma3.jinja").read()

    sysmsg = None
    if args.fewshot:
        sysmsg, _ = eval_fewshot_system()

    prompts = []
    for r in rows:
        msgs = []
        if sysmsg:
            msgs.append({"role": "system", "content": sysmsg})
        msgs.append({"role": "user",
                     "content": PROMPT_TEMPLATE.format(prompt=r["question"])})
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, chat_template=ct))

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util,
              max_model_len=args.max_model_len, dtype="bfloat16", seed=args.seed,
              enable_prefix_caching=True)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=None,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    n_correct = 0
    results = []
    for r, o in zip(rows, outs):
        target = r.get("target")
        if target is None:
            target = r["answer"].split("####")[-1].strip()
        cands = []
        for c in o.outputs:
            txt = c.text.strip()
            ok = score(txt, target)
            cands.append({"text": txt, "correct": bool(ok),
                          "ntok": len(c.token_ids)})
        n_correct += int(cands[0]["correct"])
        results.append({"question": r["question"], "target": target,
                        "cands": cands})

    pass1 = n_correct / max(1, len(results))
    anyc = sum(1 for r in results if any(c["correct"] for c in r["cands"])) / max(1, len(results))
    print(f"\nRESULT model={args.model} n={args.n} T={args.temperature} "
          f"samples={len(results)} first_sample_acc={pass1:.4f} pass@{args.n}={anyc:.4f}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
