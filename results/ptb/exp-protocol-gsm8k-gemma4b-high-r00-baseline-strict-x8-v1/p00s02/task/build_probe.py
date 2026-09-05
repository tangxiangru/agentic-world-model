"""Hold out a probe set of GSM8K TRAIN problems (never the test split).

The probe gives a cheap local dev signal (offline vLLM, zero-shot, greedy) so
checkpoints can be compared without spending the official protocol's minutes.
Any training row containing a held-out problem verbatim is removed.
"""
import argparse
import json
import random
import re

from datasets import load_from_disk


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--train-in", default="/home/ben/task/data/sft_omi2_gsm8k.jsonl")
    ap.add_argument("--train-out", default="/home/ben/task/data/sft_omi2_gsm8k_clean.jsonl")
    ap.add_argument("--probe-out", default="/home/ben/task/data/probe_gsm8ktrain300.jsonl")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    ds = load_from_disk("/home/ben/task/data/gsm8k_hf")["train"]
    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)

    probe, held = [], []
    for i in idx[: args.n]:
        q = ds[i]["question"]
        a = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
        probe.append({"id": f"gsm8ktrain-{i}", "question": q, "gold": a})
        held.append(norm(q))

    with open(args.probe_out, "w") as f:
        for r in probe:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # cheap prefilter: index held-out questions by a 48-char signature
    sigs = {}
    for h in held:
        sigs.setdefault(h[:48], []).append(h)

    kept = dropped = 0
    with open(args.train_in) as fi, open(args.train_out, "w") as fo:
        for line in fi:
            body = norm(json.loads(line)["prompt"])
            hit = any(sig in body for sig in sigs)
            if hit:
                dropped += 1
                continue
            kept += 1
            fo.write(line)
    print(f"probe {len(probe)} items -> {args.probe_out}")
    print(f"train kept {kept}, dropped {dropped} -> {args.train_out}")


if __name__ == "__main__":
    main()
