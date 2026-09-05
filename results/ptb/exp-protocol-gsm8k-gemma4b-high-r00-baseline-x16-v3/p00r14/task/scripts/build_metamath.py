"""MetaMathQA's GSM-derived subsets, reformatted to the grader's answer marker.

MetaMathQA responses end with "The answer is: N", and some carry a \\boxed{}
as well - two markers in one target is pitfall double_answer_format. Both are
removed here and a single "ANSWER: N" line is appended. Only GSM_* subsets are
used (rephrasings and self-verification variants of GSM8K *train* problems);
MATH_* rows are dropped because the benchmark is grade-school word problems.
"""
import argparse
import json
import os
import re

from datasets import load_dataset

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
ANS_TAIL = re.compile(r"\n?The answer is:?\s*(.+?)\s*$", re.S)
INT_RE = re.compile(r"^-?\d+$")


def clean_int(s):
    s = str(s).strip().replace(",", "").replace("$", "").rstrip(".")
    if s.endswith(".0"):
        s = s[:-2]
    return s if INT_RE.match(s) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=120000)
    ap.add_argument("--types", default="GSM_Rephrased,GSM_SV,GSM_FOBAR")
    ap.add_argument("--out", default=os.path.join(TASK_DIR, "data", "metamath_gsm.jsonl"))
    args = ap.parse_args()

    keep_types = set(args.types.split(","))
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    holdout = {json.loads(l)["question"]
               for l in open(os.path.join(TASK_DIR, "data", "dev_heldout250.jsonl"))}

    out, seen, n_seen_type, n_bad = [], set(), 0, 0
    for r in ds:
        if len(out) >= args.max_rows:
            break
        if r["type"] not in keep_types:
            continue
        n_seen_type += 1
        q = r["query"].strip()
        if q in seen or q in holdout or r.get("original_question", "").strip() in holdout:
            continue
        body = r["response"]
        m = ANS_TAIL.search(body)
        if not m:
            n_bad += 1
            continue
        gold = clean_int(BOXED.sub(r"\1", m.group(1)))
        if gold is None:
            n_bad += 1
            continue
        reasoning = BOXED.sub(r"\1", body[: m.start()]).strip()
        # MetaMath keeps gsm8k's own "#### N" line in some AnsAug rows.
        reasoning = re.sub(r"\n?####.*$", "", reasoning).strip()
        if not reasoning or re.search(r"\bANSWER\s*:", reasoning, re.I):
            n_bad += 1
            continue
        seen.add(q)
        out.append({"question": q, "target": f"{reasoning}\n\nANSWER: {gold}",
                    "answer": gold, "src": "metamath:" + r["type"]})

    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"types seen {n_seen_type}, rejected {n_bad}, wrote {len(out)} -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
