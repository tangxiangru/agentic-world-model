#!/usr/bin/env python3
"""Build a Python-focused SFT dataset from Magicoder (OSS-Instruct + Evol-Instruct).

Outputs:
  data/train_raw.jsonl  -- {instruction, response, text}  (text = instruction+response, for contamination check)
Run contamination_check on it, then build_final.py drops flagged lines.
"""
import json, os, re, random, argparse

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
from datasets import load_dataset

random.seed(0)

PY_FENCE = re.compile(r"```python\b", re.IGNORECASE)
ANY_FENCE = re.compile(r"```(\w+)")

def has_python_block(resp: str) -> bool:
    return bool(PY_FENCE.search(resp)) and "def " in resp

def clean_ok(instr: str, resp: str, min_len=40, max_len=6000) -> bool:
    if not instr or not resp:
        return False
    if not has_python_block(resp):
        return False
    if len(resp) < min_len or len(resp) > max_len:
        return False
    if len(instr) > 4000:
        return False
    # avoid answers dominated by a non-python language fence
    langs = [m.lower() for m in ANY_FENCE.findall(resp)]
    non_py = [l for l in langs if l not in ("python", "py", "")]
    if non_py:
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-oss", type=int, default=14000)
    ap.add_argument("--n-evol", type=int, default=10000)
    ap.add_argument("--out", default="data/train_raw.jsonl")
    args = ap.parse_args()
    os.makedirs("data", exist_ok=True)

    rows = []

    oss = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    oss_py = [r for r in oss if r["lang"] == "python"]
    random.shuffle(oss_py)
    kept = 0
    for r in oss_py:
        instr, resp = r["problem"], r["solution"]
        if clean_ok(instr, resp):
            rows.append({"instruction": instr, "response": resp, "src": "oss"})
            kept += 1
        if kept >= args.n_oss:
            break
    print(f"OSS python kept {kept} of {len(oss_py)} python rows")

    evol = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
    idx = list(range(len(evol)))
    random.shuffle(idx)
    kept = 0
    for i in idx:
        r = evol[i]
        instr, resp = r["instruction"], r["response"]
        if clean_ok(instr, resp):
            rows.append({"instruction": instr, "response": resp, "src": "evol"})
            kept += 1
        if kept >= args.n_evol:
            break
    print(f"Evol python kept {kept}")

    random.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            r["text"] = r["instruction"] + "\n\n" + r["response"]
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")

if __name__ == "__main__":
    main()
