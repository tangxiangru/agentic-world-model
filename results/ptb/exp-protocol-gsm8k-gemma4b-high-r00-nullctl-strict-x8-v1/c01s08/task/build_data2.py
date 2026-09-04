#!/usr/bin/env python3
"""Round-2 SFT data: full OpenMathInstruct-2 GSM8K slice + GSM8K train GT +
(optionally) rejection-sampled on-policy solutions."""
from __future__ import annotations
import argparse, json, os, random, re, collections
from datasets import load_dataset
from build_data import (MATH_PROMPT_TEMPLATE, strip_boxed, is_numeric, norm_num,
                        clean_gt, make_fewshot_block)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft2.jsonl")
    ap.add_argument("--omi", default="data/omi_gsm_full.jsonl")
    ap.add_argument("--omi-extra", default="data/omi_gsm8k_extra.jsonl")
    ap.add_argument("--gsm8k-per-problem", type=int, default=6)
    ap.add_argument("--rft", default="data/rft_raw.jsonl")
    ap.add_argument("--rft-per-problem", type=int, default=2)
    ap.add_argument("--omi-per-problem", type=int, default=2)
    ap.add_argument("--max-total", type=int, default=250000)
    ap.add_argument("--n-aug", type=int, default=70000)
    ap.add_argument("--fewshot-frac", type=float, default=0.14)
    ap.add_argument("--gt-repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    gsm = load_dataset("openai/gsm8k", "main")["train"]
    fs_pool = []
    for r in gsm:
        body, _, tgt = r["answer"].partition("####")
        fs_pool.append((r["question"], body.strip(), tgt.strip()))

    records = []
    for _ in range(args.gt_repeat):
        for r in gsm:
            body, tgt = clean_gt(r["answer"])
            if body and is_numeric(tgt):
                records.append((r["question"].strip(), body, norm_num(tgt), "gt"))
    print("gt:", len(records))

    seen = collections.Counter()
    aug_recs = []
    dedup = set()
    n_omi = collections.Counter()
    caps = {"gsm8k": args.gsm8k_per_problem, "augmented_gsm8k": args.omi_per_problem}
    for path in [p for p in (args.omi_extra, args.omi) if p and os.path.exists(p)]:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                q = d["problem"]
                if seen[q] >= caps.get(d["src"], args.omi_per_problem):
                    continue
                sol = strip_boxed(d["solution"]).strip()
                if not sol or len(sol) > 4000:
                    continue
                key = (q, sol)
                if key in dedup:
                    continue
                dedup.add(key)
                seen[q] += 1
                rec = (q, sol, norm_num(d["answer"]), "omi_" + d["src"])
                (aug_recs if d["src"] == "augmented_gsm8k" else records).append(rec)
                n_omi[d["src"]] += 1
    rng.shuffle(aug_recs)
    records += aug_recs[:args.n_aug]
    print("omi:", dict(n_omi), "aug kept:", min(len(aug_recs), args.n_aug))

    n_rft = 0
    if args.rft and os.path.exists(args.rft):
        with open(args.rft) as f:
            for line in f:
                d = json.loads(line)
                sols = d["solutions"]
                if not sols:
                    continue
                # keep a couple of *distinct reasoning paths* per problem
                by_path = {}
                for s in sols:
                    eqs = tuple(re.findall(r"[-+*/=]\s*-?\d[\d,.]*", s))
                    by_path.setdefault(eqs, s)
                sols = sorted(by_path.values(), key=len)
                rng.shuffle(sols)
                for s in sols[:args.rft_per_problem]:
                    s = re.sub(r"\n*ANSWER:.*$", "", s, flags=re.S).strip()
                    if not s:
                        continue
                    records.append((d["question"], s, d["answer"], "rft_" + d["tag"]))
                    n_rft += 1
    print("rft:", n_rft)

    rng.shuffle(records)
    if len(records) > args.max_total:
        records = records[:args.max_total]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    stats = collections.Counter()
    n_fs = 0
    with open(args.out, "w") as f:
        for q, sol, tgt, src in records:
            stats[src] += 1
            prompt = MATH_PROMPT_TEMPLATE.format(prompt=q.strip())
            if rng.random() < args.fewshot_frac:
                k = rng.choice([2, 4, 5, 8, 10, 10, 10])
                prompt = make_fewshot_block(rng.sample(fs_pool, k)) + "\n\n" + prompt
                n_fs += 1
            f.write(json.dumps({
                "prompt": prompt, "completion": f"{sol}\n\nANSWER: {tgt}",
                "source": src, "target": tgt,
                "text_for_decon": q.strip() + "\n" + sol + "\nANSWER: " + tgt,
            }) + "\n")
    print("total:", len(records), "fewshot:", n_fs, dict(stats), "->", args.out)


if __name__ == "__main__":
    main()
