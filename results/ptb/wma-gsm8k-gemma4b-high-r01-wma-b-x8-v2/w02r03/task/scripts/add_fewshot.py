"""Re-render a fraction of the SFT rows behind a k-shot system prefix.

The grader always sends a 10-shot system message (built from the GSM8K TRAIN
split, seed 42). Training every row zero-shot risks a train/serve mismatch, so
a slice of rows is rendered behind k demonstrations drawn from GSM8K *train*,
formatted exactly the way inspect_evals' sample_to_fewshot() formats them.
The demonstrations are random train items, never the eval's own ten.
"""
import argparse
import glob
import json
import random
import sys

sys.path.insert(0, "/home/ben/task/scripts")
import render  # noqa: E402

GSM_TRAIN = glob.glob(
    "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"
)[0]


def sample_to_fewshot(q, reasoning, target):
    # verbatim shape of inspect_evals.gsm8k.sample_to_fewshot
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="/home/ben/task/data/sft_v1.jsonl")
    ap.add_argument("--out", default="/home/ben/task/data/sft_v2.jsonl")
    ap.add_argument("--n-total", type=int, default=135000)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = render.get_tokenizer()

    import pyarrow.parquet as pq

    pool = []
    for r in pq.read_table(GSM_TRAIN).to_pylist():
        body, _, ans = r["answer"].rpartition("####")
        pool.append((r["question"].strip(), body.strip(), ans.strip().replace(",", "")))
    print("fewshot pool:", len(pool), flush=True)

    rows = [json.loads(l) for l in open(args.inp)]
    rng.shuffle(rows)
    rows = rows[: args.n_total]

    n_fs = 0
    kdist = []
    with open(args.out, "w") as fh:
        for r in rows:
            sysmsg = None
            if rng.random() < args.fewshot_frac:
                k = rng.choices([1, 2, 3, 5, 10], weights=[30, 30, 20, 12, 8])[0]
                shots = rng.sample(pool, k)
                sysmsg = "\n\n".join(sample_to_fewshot(*s) for s in shots)
                n_fs += 1
                kdist.append(k)
            r["prompt"] = render.render_prompt(tok, r["question"], system=sysmsg)
            r["nshot"] = 0 if sysmsg is None else len(kdist and [kdist[-1]] or [0]) and kdist[-1]
            fh.write(json.dumps(r) + "\n")
    from collections import Counter

    print("wrote", len(rows), "->", args.out, "fewshot rows:", n_fs, Counter(kdist))


if __name__ == "__main__":
    main()
