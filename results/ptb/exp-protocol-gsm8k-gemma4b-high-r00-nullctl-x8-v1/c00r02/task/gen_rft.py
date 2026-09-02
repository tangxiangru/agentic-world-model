#!/usr/bin/env python3
"""Rejection-sampling (STaR/RFT) data generation with the SFT model via vLLM.

Samples k solutions per training problem, keeps the ones whose final answer
matches the reference answer, and writes them as new SFT records.
Problems come from GSM8K *train* and OpenMathInstruct-2 augmented-gsm8k
problems (both allowed for training); the GSM8K test set is never touched.
"""
import argparse, json, os, random, re, hashlib
from collections import defaultdict

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

PUNCT = str.maketrans({c: None for c in "$,%"})


from inspect_ai.scorer._common import match_str


def is_correct(text: str, gold: str) -> bool:
    """Use the exact grader from the benchmark scorer."""
    try:
        _, ok = match_str(value=text, target=gold, location="end", numeric=True)
        return bool(ok)
    except Exception:
        return False


def norm(x):
    try:
        return float(str(x).translate(PUNCT))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="work/rft_data.jsonl")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n-aug", type=int, default=40000)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    random.seed(args.seed)

    # ---------------- problems ----------------
    from datasets import load_dataset, load_from_disk
    problems = []  # (question, gold)
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for r in gsm:
        problems.append((r["question"], r["answer"].split("####")[-1].strip()))
    n_gsm = len(problems)

    if args.n_aug > 0:
        tbl = load_from_disk("work/data/omi2_1M").data.table
        src = tbl.column("problem_source").to_pylist()
        probs = tbl.column("problem").to_pylist()
        answers = tbl.column("expected_answer").to_pylist()
        seen = {}
        for i in range(len(src)):
            if src[i] == "augmented_gsm8k" and probs[i] not in seen:
                seen[probs[i]] = answers[i]
        aug = [(q, a) for q, a in seen.items() if norm(a) is not None]
        random.shuffle(aug)
        problems += aug[: args.n_aug]
    print(f"{n_gsm} gsm8k-train + {len(problems)-n_gsm} augmented = {len(problems)} problems",
          flush=True)

    # ---------------- generate ----------------
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    chat_template = open("templates/gemma3.jinja").read()

    prompts = []
    for q, _ in problems:
        msgs = [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}]
        prompts.append(tok.apply_chat_template(
            msgs, chat_template=chat_template, tokenize=False, add_generation_prompt=True))

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              max_model_len=2048, enforce_eager=False, dtype="bfloat16",
              disable_log_stats=True)
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=0.95, top_k=64,
                        max_tokens=args.max_tokens, seed=args.seed,
                        stop=["<end_of_turn>"])
    outs = llm.generate(prompts, sp)

    # ---------------- filter ----------------
    records = []
    n_solved = 0
    per_problem_correct = []
    for (q, gold), o in zip(problems, outs):
        cands = []
        for c in o.outputs:
            t = c.text.strip()
            if not t or "ANSWER:" not in t:
                continue
            if is_correct(t, gold):
                cands.append(t)
        per_problem_correct.append(len(cands))
        if not cands:
            continue
        n_solved += 1
        # dedupe by normalized text, prefer shorter solutions
        uniq, seen_h = [], set()
        for t in sorted(cands, key=len):
            h = hashlib.md5(re.sub(r"\s+", " ", t).encode()).hexdigest()
            if h in seen_h:
                continue
            seen_h.add(h)
            uniq.append(t)
        for t in uniq[: args.keep_per_problem]:
            # strip the trailing ANSWER line; it is re-added by the trainer
            m = list(re.finditer(r"\n\s*ANSWER:", t))
            body = t[: m[-1].start()].strip() if m else t
            if not body:
                continue
            records.append({"question": q, "solution": body, "answer": gold,
                            "src": "rft"})

    import statistics
    print(f"solved {n_solved}/{len(problems)} problems "
          f"({n_solved/len(problems):.1%}); mean correct/k = "
          f"{statistics.mean(per_problem_correct):.2f}", flush=True)
    random.shuffle(records)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
