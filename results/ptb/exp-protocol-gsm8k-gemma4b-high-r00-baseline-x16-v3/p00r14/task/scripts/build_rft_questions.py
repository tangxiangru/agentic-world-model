"""Question pool for rejection sampling: GSM8K train, plus OpenMathInstruct-2
GSM problems that build_sft_data.py did not already use.

Only questions with a gold integer answer are kept, because the RFT filter needs
something to check the sampled solution against. Nothing here touches the test
split.
"""
import argparse
import json
import os
import random
import re

from datasets import load_dataset, load_from_disk

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INT_RE = re.compile(r"^-?\d+$")


def clean_int(s):
    s = str(s).strip().replace(",", "").replace("$", "")
    if s.endswith(".0"):
        s = s[:-2]
    return s if INT_RE.match(s) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-data", default=os.path.join(TASK_DIR, "data", "sft_v2.jsonl"))
    ap.add_argument("--omi-extra", type=int, default=30000)
    ap.add_argument("--allow-reused-omi", action="store_true",
                    help="also sample OMI problems the SFT corpus already used; "
                         "the model's own correct solutions for them are still new data")
    ap.add_argument("--out", default=os.path.join(TASK_DIR, "data", "rft_questions.jsonl"))
    args = ap.parse_args()

    used = set()
    with open(args.sft_data) as f:
        for line in f:
            used.add(json.loads(line)["question"])
    holdout = {json.loads(l)["question"]
               for l in open(os.path.join(TASK_DIR, "data", "dev_heldout250.jsonl"))}
    print(f"already in sft: {len(used)} unique questions; holdout {len(holdout)}",
          flush=True)

    out = []
    train = load_dataset("openai/gsm8k", "main", split="train")
    for i, r in enumerate(train):
        q = r["question"].strip()
        if q in holdout:
            continue
        gold = clean_int(r["answer"].rpartition("####")[2])
        if gold is not None:
            out.append({"id": f"gsm8ktrain-{i}", "question": q, "gold": gold})
    n_gsm = len(out)

    ds = load_from_disk(os.path.join(TASK_DIR, "data", "omi2_gsm"))
    order = list(range(len(ds)))
    random.Random(7).shuffle(order)
    seen = set()
    added = 0
    for i in order:
        if added >= args.omi_extra:
            break
        r = ds[i]
        q = r["problem"].strip()
        if (q in used and not args.allow_reused_omi) or q in seen or q in holdout:
            continue
        gold = clean_int(r["expected_answer"])
        if gold is None:
            continue
        seen.add(q)
        out.append({"id": f"omi-{i}", "question": q, "gold": gold})
        added += 1

    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"gsm8k train {n_gsm} + omi extra {added} = {len(out)} -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
