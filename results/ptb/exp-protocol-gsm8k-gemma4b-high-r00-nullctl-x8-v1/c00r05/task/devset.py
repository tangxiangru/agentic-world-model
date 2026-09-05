#!/usr/bin/env python3
"""Build a held-out dev set of GSM8K-style problems never used in training,
and evaluate a model on it under 0-shot and 10-shot (eval-matched) prompts."""
import argparse, glob, json, os, random, re
import pandas as pd

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text):
    idx = text.rfind("ANSWER:")
    if idx == -1:
        # fall back to last number, like the real scorer
        ms = NUM_RE.findall(text)
        if not ms:
            return None
        s = ms[-1]
    else:
        m = NUM_RE.search(text[idx + 7:])
        if not m:
            return None
        s = m.group(0)
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def build_dev(path, n=1500, seed=123):
    used = set()
    for f in ["data/sft_v1.jsonl", "data/rft_sft.jsonl", "data/sft_v2.jsonl"]:
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            for line in fh:
                used.add(json.loads(line)["problem"])
    with open("data/rft_raw.jsonl") as fh:
        for line in fh:
            used.add(json.loads(line)["problem"])
    print("used problems", len(used))
    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"))
    out, seen = [], set()
    for f in files:
        df = pd.read_parquet(f, columns=["problem", "expected_answer", "problem_source"])
        for prob, ans, src in df.itertuples(index=False):
            if src != "augmented_gsm8k" or prob in used or prob in seen:
                continue
            a = ans.strip().replace(",", "")
            if not a.lstrip("-").isdigit():
                continue
            seen.add(prob)
            out.append({"problem": prob, "answer": float(a)})
        if len(out) >= n * 3:
            break
    random.Random(seed).shuffle(out)
    out = out[:n]
    with open(path, "w") as fo:
        for r in out:
            fo.write(json.dumps(r) + "\n")
    print("dev set", len(out), "->", path)


def fewshot_block(k=10, seed=42):
    """Reproduce the exact fewshot system message the real eval uses."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=seed).select(range(k))
    parts = []
    for r in ds:
        p = r["answer"].split("####")
        parts.append(f"{r['question']}\n\nReasoning:\n" + "####".join(p[:-1]).strip()
                     + f"\n\nANSWER: {p[-1].strip()}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--model")
    ap.add_argument("--modes", default="0,10")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.build:
        build_dev(args.dev, args.n)
        if not args.model:
            return

    dev = [json.loads(l) for l in open(args.dev)]
    modes = [int(x) for x in args.modes.split(",")]
    blocks = {0: None}
    for m in modes:
        if m > 0:
            blocks[m] = fewshot_block(m)

    prompts, meta = [], []
    for m in modes:
        for r in dev:
            u = MATH_PROMPT_TEMPLATE.format(prompt=r["problem"].strip())
            if blocks[m]:
                u = blocks[m] + "\n\n" + u
            prompts.append("<bos><start_of_turn>user\n" + u + "<end_of_turn>\n<start_of_turn>model\n")
            meta.append((m, r["answer"]))

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              max_model_len=4096, enable_prefix_caching=True, dtype="bfloat16")
    sp = SamplingParams(temperature=args.temp, max_tokens=768, stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    import collections
    ok = collections.Counter(); tot = collections.Counter()
    for (m, a), o in zip(meta, outs):
        p = extract_answer(o.outputs[0].text)
        tot[m] += 1
        if p is not None and abs(p - a) < 1e-6:
            ok[m] += 1
    res = {f"{m}shot": ok[m] / tot[m] for m in modes}
    print("DEVRESULT", args.tag, json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
