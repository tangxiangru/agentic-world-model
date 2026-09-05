#!/usr/bin/env python3
"""Rejection-sampling data generation: sample k CoTs per training question with vLLM,
keep the ones whose final answer matches the gold answer, dedupe by reasoning skeleton."""
from __future__ import annotations

import argparse
import json
import os
import random
import re

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/home/ben/hf_cache/hub")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def norm_num(s):
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(round(v, 5))


def extract_answer(text: str):
    idx = text.rfind("ANSWER:")
    if idx < 0:
        return None
    tail = text[idx + 7:]
    m = NUM_RE.search(tail)
    if not m:
        return None
    return norm_num(m.group(0))


def skeleton(sol: str) -> str:
    """Dedup key: the sequence of numbers appearing in the solution."""
    return "|".join(NUM_RE.findall(sol))


def load_jsonl(path, limit=None):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-per-q", type=int, default=4)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = open("templates/gemma3.jinja").read()

    records = []
    for spec in args.data:
        if ":" in spec:
            path, lim = spec.rsplit(":", 1)
            lim = int(lim)
        else:
            path, lim = spec, None
        rs = load_jsonl(path, lim)
        print(f"{path}: {len(rs)}")
        records += rs
    print("questions:", len(records))

    from vllm.inputs import TokensPrompt

    texts_in = []
    for r in records:
        msgs = [{"role": "user", "content": PROMPT_TEMPLATE.format(prompt=r["question"])}]
        texts_in.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    # the chat template already emits <bos>; tokenize without adding another one
    enc = tok(texts_in, add_special_tokens=False)["input_ids"]
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in enc]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=1280,
              enable_prefix_caching=False, dtype="bfloat16", max_num_seqs=512,
              max_num_batched_tokens=16384, disable_log_stats=True)
    # NOTE: with n>1 vLLM fans out child requests from the *pre*-generation-config
    # SamplingParams, so the extra eos id (<end_of_turn>=106) is lost. Set it explicitly.
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=0.95, max_tokens=args.max_tokens,
                        seed=1234, detokenize=False, stop_token_ids=[eot])
    rng = random.Random(0)
    n_correct = n_total = solved_q = n_kept = n_q = 0
    CHUNK = 3000
    fout = open(args.out, "w")
    for start in range(0, len(records), CHUNK):
        chunk_r = records[start:start + CHUNK]
        outs = llm.generate(prompts[start:start + CHUNK], sp, use_tqdm=False)
        # batch-detokenize at the end (much cheaper than incremental detokenization)
        flat = [list(c.token_ids) for o in outs for c in o.outputs]
        texts = tok.batch_decode(flat, skip_special_tokens=True)
        it = iter(texts)
        for o in outs:
            for c in o.outputs:
                c.text = next(it)

        for r, o in zip(chunk_r, outs):
            n_q += 1
            gold = norm_num(r["answer"])
            cands = []
            for c in o.outputs:
                n_total += 1
                txt = c.text.strip()
                if c.finish_reason != "stop":
                    continue
                pred = extract_answer(txt)
                if pred is None or pred != gold:
                    continue
                n_correct += 1
                # strip the trailing ANSWER line -> keep pure reasoning
                idx = txt.rfind("ANSWER:")
                sol = txt[:idx].strip()
                if sol:
                    cands.append(sol)
            if not cands:
                continue
            solved_q += 1
            by_skel = {}
            for s in cands:
                by_skel.setdefault(skeleton(s), []).append(s)
            uniq = [min(v, key=len) for v in by_skel.values()]
            rng.shuffle(uniq)
            for s in uniq[:args.max_per_q]:
                fout.write(json.dumps({"question": r["question"], "solution": s,
                                       "answer": gold, "src": "rft"}) + "\n")
                n_kept += 1
        fout.flush()
        print(f"[chunk {start//CHUNK}] q={n_q}/{len(records)} pass={n_correct/max(n_total,1):.3f} "
              f"solved={solved_q/max(n_q,1):.3f} kept={n_kept}", flush=True)
    fout.close()
    print(f"DONE pass rate {n_correct}/{n_total} = {n_correct/max(n_total,1):.3f}")
    print(f"questions solved at least once: {solved_q}/{len(records)}")
    print(f"kept solutions: {n_kept}")


if __name__ == "__main__":
    main()
