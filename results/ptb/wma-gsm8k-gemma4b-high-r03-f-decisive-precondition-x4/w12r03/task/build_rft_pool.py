"""Question pool for rejection sampling (no solutions, just question + gold).

Half original GSM8K train questions (minus the probe), half OpenMathInstruct-2
augmented_gsm8k questions that are NOT already in the SFT file, so the model
sees fresh problems as well as ones it was trained on.
"""
import argparse
import json
import re

from datasets import load_dataset


def norm_q(q):
    return re.sub(r"\s+", " ", q.strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/rft_pool.jsonl")
    ap.add_argument("--n-fresh", type=int, default=14000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    probe = {norm_q(json.loads(l)["question"]) for l in open("/home/ben/task/data/probe250.jsonl")}
    seen = set()
    rows = []

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for r in gsm:
        q = norm_q(r["question"])
        if q in probe or q in seen:
            continue
        seen.add(q)
        rows.append({"id": f"gsm-{len(rows)}", "question": r["question"],
                     "gold": r["answer"].split("####")[-1].strip().replace(",", ""),
                     "src": "gsm8k_train"})
    print(f"gsm8k train questions: {len(rows)}")

    in_sft = {norm_q(json.loads(l)["question"]) for l in open("/home/ben/task/data/sft_v1.jsonl")}
    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train")
    omi = omi.filter(lambda b: [s == "augmented_gsm8k" for s in b["problem_source"]],
                     batched=True, num_proc=16)
    omi = omi.shuffle(seed=args.seed)
    n_fresh = 0
    for r in omi:
        if n_fresh >= args.n_fresh:
            break
        q = norm_q(r["problem"])
        if q in probe or q in seen or q in in_sft:
            continue
        a = r["expected_answer"].strip().replace(",", "")
        if not (a.lstrip("-").isdigit() and len(a) <= 12):
            continue
        seen.add(q)
        n_fresh += 1
        rows.append({"id": f"aug-{n_fresh}", "question": r["problem"], "gold": a,
                     "src": "augmented_gsm8k_fresh"})
    print(f"fresh augmented questions: {n_fresh}")

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
