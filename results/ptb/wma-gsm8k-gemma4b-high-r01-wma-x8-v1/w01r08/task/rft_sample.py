#!/usr/bin/env python3
"""Rejection-sampling data: sample k solutions per problem from a checkpoint with
vLLM, keep the ones whose LAST number equals the gold answer (the grader's rule).

Problems come from the GSM8K TRAIN split and/or OpenMathInstruct-2 problems, never
from the test split.
"""
from __future__ import annotations
import argparse, json, random, re, os, collections

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()
STOP = "<end_of_turn>"


def last_number(t: str):
    ns = re.findall(r"-?\d[\d,]*\.?\d*", t)
    if not ns:
        return None
    return ns[-1].replace(",", "").rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--n-problems", type=int, default=20000)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default="both", choices=["gsm8k", "omi2", "both"])
    ap.add_argument("--fewshot-frac", type=float, default=0.20)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    problems = []
    if args.source in ("gsm8k", "both"):
        from datasets import load_dataset
        for r in load_dataset("openai/gsm8k", "main")["train"]:
            problems.append((r["question"].strip(), r["answer"].split("####")[-1].strip()))
    if args.source in ("omi2", "both"):
        import glob, pyarrow.parquet as pq
        seen = set(p for p, _ in problems)
        want = args.n_problems - len(problems)
        for f in sorted(glob.glob("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
                                  "snapshots/*/data/*.parquet")):
            if want <= 0:
                break
            for b in pq.ParquetFile(f).iter_batches(batch_size=20000):
                d = b.to_pydict()
                for prob, ans, src in zip(d["problem"], d["expected_answer"], d["problem_source"]):
                    if want <= 0:
                        break
                    if src not in ("gsm8k", "augmented_gsm8k"):
                        continue
                    if not re.match(r"^-?\d+(\.\d+)?$", ans.strip()):
                        continue
                    p = prob.strip()
                    if p in seen or len(p) > 2000:
                        continue
                    seen.add(p)
                    problems.append((p, ans.strip()))
                    want -= 1
                if want <= 0:
                    break
    rng.shuffle(problems)
    problems = problems[:args.n_problems]
    print(f"{len(problems)} problems", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tpl = open("templates/gemma3.jinja").read()
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=p)}],
        chat_template=tpl, tokenize=False, add_generation_prompt=True) for p, _ in problems]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=2048,
              dtype="bfloat16", seed=args.seed, enable_prefix_caching=True)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[106, 1], seed=args.seed)
    outs = llm.generate(prompts, sp)

    kept, n_corr, n_tot = [], 0, 0
    exact10 = open("data/fewshot_system.txt").read()
    pool = None
    for (prob, gold), o in zip(problems, outs):
        cands = []
        for c in o.outputs:
            t = c.text.strip()
            n_tot += 1
            if last_number(t) != gold.replace(",", ""):
                continue
            if not re.search(r"ANSWER:\s*\$?-?[\d,]*\.?\d*\s*$", t):
                continue
            n_corr += 1
            cands.append(t)
        # dedup then keep the shortest few (short correct chains are cleaner)
        cands = sorted(set(cands), key=len)[:args.keep_per_problem]
        for t in cands:
            kept.append({"problem": prob, "completion": t + STOP, "answer": gold,
                         "source": "synthetic:self"})
    print(f"sampled {n_tot}, correct {n_corr} ({n_corr/max(1,n_tot):.3f}), kept {len(kept)}",
          flush=True)

    rng.shuffle(kept)
    with open(args.out, "w") as f:
        for r in kept:
            system = None
            u = rng.random()
            if u < args.fewshot_frac * 0.25:
                system = exact10
            elif u < args.fewshot_frac:
                if pool is None:
                    from datasets import load_dataset
                    pool = []
                    for x in load_dataset("openai/gsm8k", "main")["train"]:
                        q = x["question"]
                        a = x["answer"].split("####")
                        tg = a.pop().strip()
                        pool.append(f"{q}\n\nReasoning:\n{'####'.join(a).strip()}\n\nANSWER: {tg}")
                system = "\n\n".join(rng.sample(pool, rng.choice([2, 3, 4])))
            f.write(json.dumps({"system": system,
                                "prompt_user": MATH_PROMPT_TEMPLATE.format(prompt=r["problem"]),
                                "completion": r["completion"], "answer": r["answer"],
                                "source": r["source"]}) + "\n")
    print("wrote", args.out, len(kept))
    ck = args.out.replace(".jsonl", "_forcheck.jsonl")
    with open(ck, "w") as f:
        for r in kept:
            f.write(json.dumps({"text": r["problem"] + "\n" +
                                r["completion"].replace(STOP, "")}) + "\n")
    print("wrote", ck)


if __name__ == "__main__":
    main()
