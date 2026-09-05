#!/usr/bin/env python3
"""Turn gen.py's k-sample output into an RFT training file.

Keeps only samples whose final ANSWER equals the gold answer, deduplicates the
distinct reasoning paths per problem, and spends the per-problem budget on the
problems the model finds hard (low pass rate) rather than the ones it already
solves every time. Output is written in exactly the same prompt/completion
rendering as the SFT files, so train_sft.py's asserts apply unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter

from transformers import AutoTokenizer

import common

WS = re.compile(r"\s+")


def signature(text: str) -> str:
    """Coarse fingerprint: the sequence of numbers in the chain."""
    return " ".join(re.findall(r"-?\d+\.?\d*", text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="gen.py output jsonl")
    ap.add_argument("--out", default="data/rft_v1.jsonl")
    ap.add_argument("--raw-out", default="data/rft_v1_raw.jsonl")
    ap.add_argument("--mix", default=None, help="teacher jsonl to mix in (prompt/completion)")
    ap.add_argument("--mix-raw", default=None,
                    help="the question/answer file paired line-for-line with --mix")
    ap.add_argument("--mix-n", type=int, default=0, help="random teacher rows to mix in")
    ap.add_argument("--mix-unsolved", action="store_true",
                    help="also mix in every teacher row whose problem this model solved 0/k times")
    ap.add_argument("--max-seq-len", type=int, default=2816)
    ap.add_argument("--max-completion-tokens", type=int, default=640)
    ap.add_argument("--easy-cap", type=int, default=1, help="kept solutions when pass rate is high")
    ap.add_argument("--hard-cap", type=int, default=3, help="kept solutions when pass rate is low")
    ap.add_argument("--hard-threshold", type=float, default=0.5)
    ap.add_argument("--p-full-fewshot", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(common.BASE_SNAPSHOT)
    full_fewshot = common.fewshot_system_message()

    stats = Counter()
    kept_rows = []
    unsolved: set[str] = set()
    for line in open(args.samples):
        r = json.loads(line)
        cands = r["candidates"]
        n_ok = sum(c["correct"] for c in cands)
        stats["problems"] += 1
        if n_ok == 0:
            stats["unsolved"] += 1
            unsolved.add(r["question"].strip())
            continue
        rate = n_ok / len(cands)
        cap = args.hard_cap if rate <= args.hard_threshold else args.easy_cap
        if rate <= args.hard_threshold:
            stats["hard"] += 1

        good, sigs = [], set()
        for c in sorted((c for c in cands if c["correct"]), key=lambda c: len(c["text"])):
            body = c["text"].strip()
            if not body or "ANSWER:" not in body:
                continue
            body = body[: body.rindex("ANSWER:")].strip()
            if not body or "####" in body:
                continue
            s = signature(body)
            if s in sigs:
                continue
            sigs.add(s)
            good.append(body)
            if len(good) >= cap:
                break
        for body in good:
            kept_rows.append({"question": r["question"], "body": body, "answer": r["answer"]})
        stats["kept"] += len(good)

    print("sampling stats:", dict(stats),
          f"solve_rate={1 - stats['unsolved']/max(1,stats['problems']):.3f}")
    rng.shuffle(kept_rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lens, n_drop = [], 0
    with open(args.out, "w") as f, open(args.raw_out, "w") as fr:
        for c in kept_rows:
            completion = common.format_target(c["body"], c["answer"])
            ctoks = tok.encode(completion, add_special_tokens=False)
            if len(ctoks) > args.max_completion_tokens:
                n_drop += 1
                continue
            sysmsg = full_fewshot if rng.random() < args.p_full_fewshot else None
            messages = ([{"role": "system", "content": sysmsg}] if sysmsg else [])
            messages.append({"role": "user", "content": common.user_message(c["question"])})
            prompt = tok.apply_chat_template(
                messages, chat_template=common.chat_template(),
                tokenize=False, add_generation_prompt=True)
            total = len(tok.encode(prompt, add_special_tokens=False)) + len(ctoks)
            if total > args.max_seq_len:
                n_drop += 1
                continue
            f.write(json.dumps({"prompt": prompt, "completion": completion,
                                "kind": "rft", "n_tokens": total}) + "\n")
            fr.write(json.dumps({"question": c["question"],
                                 "answer": c["body"] + f"\n\nANSWER: {c['answer']}"}) + "\n")
            lens.append(total)

    n_rft = len(lens)
    if args.mix:
        pool = [json.loads(l) for l in open(args.mix)]
        raw = [json.loads(l) for l in open(args.mix_raw)] if args.mix_raw else None
        chosen, taken = [], set()
        if args.mix_unsolved and raw is not None:
            assert len(raw) == len(pool), (len(raw), len(pool))
            for i, q in enumerate(raw):
                if q["question"].strip() in unsolved:
                    chosen.append(pool[i])
                    taken.add(i)
            print(f"mixed in {len(chosen)} teacher rows for problems solved 0/{'k'} times")
        rest = [i for i in range(len(pool)) if i not in taken]
        rng.shuffle(rest)
        chosen += [pool[i] for i in rest[: args.mix_n]]
        with open(args.out, "a") as f:
            for r in chosen:
                f.write(json.dumps(r) + "\n")
                lens.append(r["n_tokens"])
        print(f"mixed in {len(chosen)} teacher rows total from {args.mix}")

    lens.sort()
    print(f"wrote {n_rft} rft rows (+{len(lens)-n_rft} mixed) to {args.out}, dropped {n_drop}")
    print(f"token len p50={lens[len(lens)//2]} max={lens[-1]} total={sum(lens)/1e6:.2f}M")


if __name__ == "__main__":
    main()
