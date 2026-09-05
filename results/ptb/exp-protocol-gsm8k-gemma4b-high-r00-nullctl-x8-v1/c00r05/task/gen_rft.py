#!/usr/bin/env python3
"""Rejection sampling: generate solutions with the SFT model, keep the correct ones."""
import argparse, glob, json, os, random, re
import pandas as pd

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def build_prompt(q, shots=None):
    u = MATH_PROMPT_TEMPLATE.format(prompt=q.strip())
    if shots:
        block = "\n\n".join(f"{a}\n\nReasoning:\n{b}\n\nANSWER: {c}" for a, b, c in shots)
        u = block + "\n\n" + u
    return "<bos><start_of_turn>user\n" + u + "<end_of_turn>\n<start_of_turn>model\n"


NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text):
    idx = text.rfind("ANSWER:")
    if idx == -1:
        return None
    tail = text[idx + 7:].strip()
    m = NUM_RE.search(tail)
    if not m:
        return None
    s = m.group(0).replace(",", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v


def eq_signature(text):
    """Dedupe key: the multiset of arithmetic expressions/numbers used."""
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return tuple(n.replace(",", "") for n in nums)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/rft_raw.jsonl")
    ap.add_argument("--k-gsm", type=int, default=8)
    ap.add_argument("--k-aug", type=int, default=2)
    ap.add_argument("--n-aug", type=int, default=20000)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--fewshot-frac", type=float, default=0.25)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    fs_pool = []
    problems = []  # (qid, question, answer_float, source)
    for i, r in enumerate(gsm):
        parts = r["answer"].split("####")
        tgt = parts[-1].strip().replace(",", "")
        fs_pool.append((r["question"], "####".join(parts[:-1]).strip(), tgt))
        problems.append((f"g{i}", r["question"], float(tgt), "gsm8k_train", args.k_gsm))

    # augmented gsm8k problems from OpenMathInstruct-2
    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"))
    seen = set()
    aug = []
    for f in files:
        df = pd.read_parquet(f, columns=["problem", "expected_answer", "problem_source"])
        for prob, ans, src in df.itertuples(index=False):
            if src != "augmented_gsm8k" or prob in seen:
                continue
            a = ans.strip().replace(",", "")
            if not (a.lstrip("-").isdigit()):
                continue
            seen.add(prob)
            aug.append((prob, float(a)))
        if len(aug) > args.n_aug * 3:
            break
    rng.shuffle(aug)
    for i, (p, a) in enumerate(aug[:args.n_aug]):
        problems.append((f"a{i}", p, a, "aug_gsm8k", args.k_aug))
    print("problems:", len(problems), flush=True)

    prompts, meta = [], []
    for qid, q, a, src, k in problems:
        for j in range(k):
            shots = None
            if rng.random() < args.fewshot_frac:
                shots = rng.sample(fs_pool, rng.randint(1, 10))
            prompts.append(build_prompt(q, shots))
            meta.append((qid, q, a, src))
    print("prompts:", len(prompts), flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              max_model_len=4096, enable_prefix_caching=True, dtype="bfloat16")
    sp = SamplingParams(temperature=args.temp, top_p=0.95, top_k=64,
                        max_tokens=args.max_tokens, seed=None,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    n_ok = 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fo:
        for (qid, q, a, src), o in zip(meta, outs):
            text = o.outputs[0].text.strip()
            pred = extract_answer(text)
            ok = pred is not None and abs(pred - a) < 1e-6
            if ok:
                n_ok += 1
            fo.write(json.dumps({"qid": qid, "problem": q, "answer": a, "source": src,
                                 "gen": text, "correct": bool(ok)}) + "\n")
    print("correct:", n_ok, "/", len(meta), flush=True)


if __name__ == "__main__":
    main()
