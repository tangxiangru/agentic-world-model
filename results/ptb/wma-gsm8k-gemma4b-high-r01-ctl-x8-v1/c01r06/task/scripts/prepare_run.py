"""Turn a built sft jsonl into (a) a held-out probe set and (b) the exact file
the trainer reads, after a contamination check.

The probe set is 200 problems from the GSM8K *train* split, removed from
training data (both the original row and any OpenMathInstruct-2 row whose
problem string is identical).  It exists so that failure analysis and the
card's watch_set never touch the benchmark's test copy.
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402


def gsm8k_train():
    rows = []
    for f in glob.glob(
        "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"
    ):
        d = pq.read_table(f).to_pydict()
        for q, a in zip(d["question"], d["answer"]):
            rows.append((q, a.split("####")[-1].strip(),
                         fmt.normalize_solution(a.split("####")[0])))
    return rows


def question_of(prompt):
    body = prompt.split("<start_of_turn>user\n", 1)[1].rsplit("<end_of_turn>", 1)[0]
    head = "(without quotes) where $ANSWER is the answer to the problem.\n\n"
    q = body.split(head, 1)[-1]
    return q.split("\n\nRemember to put your answer")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-out", default="/home/ben/task/data/probe200.jsonl")
    ap.add_argument("--probe-n", type=int, default=200)
    ap.add_argument("--n-rows", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots-out", default="/home/ben/task/data/fewshot10.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gs = gsm8k_train()
    rng.shuffle(gs)
    probe = gs[: args.probe_n]
    shots = gs[args.probe_n: args.probe_n + 10]
    banned = {q for q, _, _ in probe}

    with open(args.probe_out, "w") as f:
        for i, (q, a, r) in enumerate(probe):
            f.write(json.dumps({"id": f"probe-{i:03d}", "question": q, "gold": a}) + "\n")
    with open(args.shots_out, "w") as f:
        json.dump([fmt.fewshot_block(q, r, a) for q, a, r in shots], f)

    kept, dropped = [], 0
    with open(args.src) as fh:
        for line in fh:
            r = json.loads(line)
            if question_of(r["prompt"]) in banned:
                dropped += 1
                continue
            kept.append(line)
    print(f"probe {len(probe)} | src rows kept {len(kept)} dropped-as-probe {dropped}",
          flush=True)
    rng.shuffle(kept)
    kept = kept[: args.n_rows]
    with open(args.out, "w") as fh:
        fh.writelines(kept)
    print(f"wrote {args.out} ({len(kept)} rows)", flush=True)

    chk = args.out.replace(".jsonl", "") + "_check.jsonl"
    with open(args.out) as fh, open(chk, "w") as o:
        for line in fh:
            r = json.loads(line)
            o.write(json.dumps({
                "question": question_of(r["prompt"]),
                "answer": r["completion"].replace(fmt.STOP_TOKEN, "").strip(),
            }) + "\n")
    print(f"wrote {chk}", flush=True)


if __name__ == "__main__":
    main()
