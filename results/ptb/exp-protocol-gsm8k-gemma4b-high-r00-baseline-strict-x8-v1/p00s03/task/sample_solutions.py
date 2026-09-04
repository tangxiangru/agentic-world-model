#!/usr/bin/env python3
"""Sample k solutions per problem from a trained checkpoint (offline vLLM),
keep the ones whose ANSWER line matches the gold answer, and write them in the
same jsonl schema build_data.py produces.  This is the rejection-sampling
(RFT) data generator: the targets are the model's own correct chains.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict

import common_fmt

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")


def extract(txt: str):
    m = ANS_RE.findall(txt)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True, help="jsonl with question/answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-keep-per-problem", type=int, default=3)
    ap.add_argument("--easy-keep", type=int, default=1)
    ap.add_argument("--hard-rate", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fewshot-frac-4", type=float, default=0.10)
    ap.add_argument("--fewshot-frac-10", type=float, default=0.10)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    probs = [json.loads(l) for l in open(args.problems)]
    if args.limit:
        probs = probs[: args.limit]
    print(f"{len(probs)} problems x k={args.k}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = common_fmt.load_template()
    # NB: pass token ids, not strings.  llm.generate(str) tokenizes with
    # add_special_tokens=True, which prepends a SECOND <bos> on top of the one
    # the chat template already emits; gemma-3 degrades badly with a double bos
    # (measured: solve rate 0.52 vs 0.93 on the same problems).
    from vllm import TokensPrompt
    prompts = [
        TokensPrompt(prompt_token_ids=tok(
            tok.apply_chat_template(common_fmt.build_messages(p["question"]),
                                    tokenize=False, add_generation_prompt=True),
            add_special_tokens=False)["input_ids"])
        for p in probs
    ]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
              seed=args.seed, enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=None)
    outs = llm.generate(prompts, sp)

    kept, n_correct, n_any = [], 0, 0
    solve_hist = defaultdict(int)
    for p, o in zip(probs, outs):
        gold = float(str(p["answer"]).replace(",", ""))
        good = []
        for c in o.outputs:
            if c.finish_reason != "stop":
                continue
            v = extract(c.text)
            if v is not None and abs(v - gold) < 1e-6:
                good.append(c.text.strip())
        n_correct += len(good)
        n_any += bool(good)
        solve_hist[len(good)] += 1
        # hard problems (solved rarely) contribute more chains than easy ones
        rate = len(good) / args.k
        cap = args.max_keep_per_problem if rate <= args.hard_rate else args.easy_keep
        good = sorted(set(good), key=len)[:cap]
        for g in good:
            kept.append({"question": p["question"], "target": g if g.endswith(common_fmt.STOP_TOKEN)
                         else g + common_fmt.STOP_TOKEN, "answer": str(p["answer"]),
                         "source": "rft"})
    print("solved-count histogram (k=%d):" % args.k, dict(sorted(solve_hist.items())))
    print(f"solved at least once: {n_any}/{len(probs)} ({n_any/len(probs):.3f}); "
          f"correct samples {n_correct}/{len(probs)*args.k} ({n_correct/(len(probs)*args.k):.3f}); "
          f"kept {len(kept)}")

    # attach few-shot prefixes to the same fraction as the SFT corpus
    import pyarrow.parquet as pq
    import glob as _glob
    gt = pq.read_table(sorted(_glob.glob(
        "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"))[0]).to_pylist()
    pool = []
    for r in gt:
        body, _, a = r["answer"].rpartition("####")
        pool.append((r["question"].strip(), re.sub(r"<<[^>]*>>", "", body).strip(),
                     a.strip().replace(",", "")))
    rng = random.Random(args.seed)
    rng.shuffle(kept)
    n4 = int(len(kept) * args.fewshot_frac_4)
    n10 = int(len(kept) * args.fewshot_frac_10)
    with open(args.out, "w") as fo, open(args.out.replace(".jsonl", ".decon.jsonl"), "w") as fd:
        for i, r in enumerate(kept):
            k = 4 if i < n4 else (10 if i < n4 + n10 else 0)
            r["n_shot"] = k
            r["shots"] = [pool[j] for j in rng.sample(range(len(pool)), k)] if k else []
            fo.write(json.dumps(r) + "\n")
            fd.write(json.dumps({"question": r["question"], "answer": r["target"]}) + "\n")
    print("wrote", args.out, len(kept))


if __name__ == "__main__":
    main()
