#!/usr/bin/env python3
"""Rejection-sampling data generation: sample k solutions per training question
from a checkpoint, keep the ones whose ANSWER line matches the reference answer.

Questions come from the GSM8K *train* split and from OpenMathInstruct-2's
augmented-gsm8k problems. The benchmark test split is never touched.
"""
import argparse, collections, glob, json, random, re

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"
ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")


def norm(x: str) -> str | None:
    try:
        v = float(str(x).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(v)


def load_gsm8k_train():
    f = glob.glob("/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet")[0]
    d = pq.read_table(f).to_pydict()
    out = []
    for q, a in zip(d["question"], d["answer"]):
        out.append((q.strip(), norm(a.split("####")[-1])))
    return [(q, a) for q, a in out if a is not None]


def load_augmented(n, seed, exclude):
    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"))
    seen = {}
    for f in files:
        d = pq.read_table(f, columns=["problem", "expected_answer", "problem_source"]).to_pydict()
        for q, a, s in zip(d["problem"], d["expected_answer"], d["problem_source"]):
            if s != "augmented_gsm8k":
                continue
            q = q.strip()
            if q in seen or q in exclude or len(q) > 1200:
                continue
            v = norm(a)
            if v is not None:
                seen[q] = v
    items = sorted(seen.items())
    random.Random(seed).shuffle(items)
    return items[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-orig", type=int, default=8)
    ap.add_argument("--k-aug", type=int, default=4)
    ap.add_argument("--n-aug", type=int, default=25000)
    ap.add_argument("--keep-per-question", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-out", type=str, default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open(TEMPLATE_PATH).read()

    orig = load_gsm8k_train()
    aug = load_augmented(args.n_aug, args.seed, exclude={q for q, _ in orig}) if args.n_aug else []
    print(f"questions: {len(orig)} gsm8k-train, {len(aug)} augmented", flush=True)

    jobs = [(q, a, args.k_orig, "gsm8k") for q, a in orig] + [(q, a, args.k_aug, "augmented_gsm8k") for q, a in aug]
    prompts, meta = [], []
    for q, a, k, src in jobs:
        p = tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}],
            tokenize=False, add_generation_prompt=True)
        prompts.append(p)
        meta.append((q, a, k, src))

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=1536,
              dtype="bfloat16", enable_prefix_caching=True, seed=args.seed)
    # one request per prompt, n = k samples
    outs = []
    B = 4000
    for i in range(0, len(prompts), B):
        chunk = prompts[i:i + B]
        cmeta = meta[i:i + B]
        by_k = collections.defaultdict(list)
        for j, m in enumerate(cmeta):
            by_k[m[2]].append(j)
        res = [None] * len(chunk)
        for k, idxs in by_k.items():
            sp = SamplingParams(n=k, temperature=args.temperature, top_p=args.top_p,
                                top_k=64, max_tokens=args.max_tokens, seed=None,
                                stop_token_ids=[1, 106])  # <eos>, <end_of_turn>
            r = llm.generate([chunk[j] for j in idxs], sp)
            for j, rr in zip(idxs, r):
                res[j] = rr
        outs.extend(res)
        print(f"generated {i + len(chunk)}/{len(prompts)}", flush=True)

    rng = random.Random(args.seed)
    rows, stats = [], collections.Counter()
    for (q, gold, k, src), o in zip(meta, outs):
        cands, seen = [], set()
        for c in o.outputs:
            text = c.text.replace("<end_of_turn>", "").strip()
            stats["samples"] += 1
            if c.finish_reason != "stop":
                stats["no_stop"] += 1
                continue
            m = ANS_RE.search(text)
            if not m:
                stats["no_answer"] += 1
                continue
            if len(ANS_RE.findall(text)) != 1:
                stats["multi_answer"] += 1
                continue
            if norm(m.group(1)) != gold:
                stats["wrong"] += 1
                continue
            body = text[: m.start()].rstrip()
            key = " ".join(body.split())[:400]
            if key in seen:
                stats["dup"] += 1
                continue
            seen.add(key)
            cands.append((body, gold))
        if not cands:
            stats["q_unsolved"] += 1
            continue
        stats["q_solved"] += 1
        rng.shuffle(cands)
        for body, gold in cands[: args.keep_per_question]:
            rows.append({"question": q, "solution": body, "answer": gold,
                         "target": body + "\nANSWER: " + gold + "<end_of_turn>", "source": "rft:" + src})
            stats["kept"] += 1

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats["written"] = len(rows)
    print(json.dumps(dict(stats), indent=1))
    if args.stats_out:
        json.dump(dict(stats), open(args.stats_out, "w"), indent=1)


if __name__ == "__main__":
    main()
