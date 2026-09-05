"""Build the stage-1 SFT corpus.

Sources, both derived from the GSM8K *train* split only:
  - openai/gsm8k train, minus a 250-item holdout kept for watch sets
  - nvidia/OpenMathInstruct-2, rows whose problem_source is gsm8k or
    augmented_gsm8k (augmentations of GSM8K train problems)

Every target is reshaped into the one format the grader reads: free-form
reasoning, then a final line "ANSWER: <int>". The GSM8K calculator annotations
(<<48/2=24>>) and OpenMathInstruct's \\boxed{} are removed so the target
carries exactly one answer marker.
"""
import argparse
import json
import os
import random
import re

from datasets import load_dataset, load_from_disk

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDOUT_N = 250
HOLDOUT_SEED = 12345

CALC = re.compile(r"<<[^>]*>>")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
INT_RE = re.compile(r"^-?\d+$")


def clean_int(s):
    s = str(s).strip().replace(",", "").replace("$", "")
    if s.endswith(".0"):
        s = s[:-2]
    return s if INT_RE.match(s) else None


def strip_tail_boilerplate(sol):
    """Drop a trailing 'The answer is ...' / 'So the answer is ...' sentence."""
    lines = [ln for ln in sol.strip().split("\n")]
    while lines and re.match(
        r"^\s*(so,?\s+)?(thus,?\s+)?(therefore,?\s+)?the (final )?answer is\b.*$",
        lines[-1], re.I,
    ):
        lines.pop()
    return "\n".join(lines).strip()


def finalize(reasoning, answer):
    """Attach the single answer marker; reject anything that already has one."""
    reasoning = strip_tail_boilerplate(reasoning)
    if not reasoning:
        return None
    if re.search(r"\bANSWER\s*:", reasoning, re.I):
        return None
    return f"{reasoning}\n\nANSWER: {answer}"


def gsm8k_rows(holdout_ids):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for i, r in enumerate(ds):
        if i in holdout_ids:
            continue
        body, _, ans = r["answer"].rpartition("####")
        ans = clean_int(ans)
        if ans is None:
            continue
        reasoning = CALC.sub("", body).strip()
        tgt = finalize(reasoning, ans)
        if tgt:
            out.append({"question": r["question"].strip(), "target": tgt,
                        "answer": ans, "src": "gsm8k_train"})
    return out


def omi_rows(cap_per_problem, max_rows, rng):
    path = os.path.join(TASK_DIR, "data", "omi2_gsm")
    if os.path.isdir(path):
        ds = load_from_disk(path)
    else:
        ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M").filter(
            lambda r: r["problem_source"] in ("gsm8k", "augmented_gsm8k"), num_proc=8
        )
    seen = {}
    out = []
    order = list(range(len(ds)))
    rng.shuffle(order)
    for i in order:
        if len(out) >= max_rows:
            break
        r = ds[i]
        q = r["problem"].strip()
        if seen.get(q, 0) >= cap_per_problem:
            continue
        ans = clean_int(r["expected_answer"])
        if ans is None:
            continue
        sol = BOXED.sub(r"\1", r["generated_solution"])
        if "\\boxed" in sol:
            continue
        tgt = finalize(sol, ans)
        if tgt is None:
            continue
        seen[q] = seen.get(q, 0) + 1
        out.append({"question": q, "target": tgt, "answer": ans,
                    "src": r["problem_source"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omi-max", type=int, default=60000)
    ap.add_argument("--omi-cap-per-problem", type=int, default=1)
    ap.add_argument("--gsm-repeat", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(TASK_DIR, "data", "sft_v1.jsonl"))
    ap.add_argument("--holdout-out",
                    default=os.path.join(TASK_DIR, "data", "dev_heldout250.jsonl"))
    args = ap.parse_args()

    rng = random.Random(0)
    train_full = load_dataset("openai/gsm8k", "main", split="train")
    holdout_ids = set(random.Random(HOLDOUT_SEED).sample(range(len(train_full)),
                                                         HOLDOUT_N))
    with open(args.holdout_out, "w") as f:
        for i in sorted(holdout_ids):
            r = train_full[i]
            gold = clean_int(r["answer"].rpartition("####")[2])
            f.write(json.dumps({"id": f"train-{i}", "question": r["question"].strip(),
                                "gold": gold}) + "\n")
    print(f"holdout: {len(holdout_ids)} -> {args.holdout_out}", flush=True)

    rows = []
    g = gsm8k_rows(holdout_ids)
    print(f"gsm8k_train usable: {len(g)} (x{args.gsm_repeat})", flush=True)
    rows += g * args.gsm_repeat
    o = omi_rows(args.omi_cap_per_problem, args.omi_max, rng)
    print(f"openmathinstruct2 gsm: {len(o)}", flush=True)
    rows += o

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
