#!/usr/bin/env python3
"""Add microsoft/orca-math-word-problems-200k as a source of NEW grade-school problems.

Its answers are prose with no canonical marker, so the final number is extracted
the same way the grader extracts one (last numeric token) and only rows where
that number sits in the closing sentence are kept -- a strict filter, because a
mis-extracted label is a wrong training target.
"""
import argparse, json, os, random, re
os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
from datasets import load_dataset
from build_data import MATH_PROMPT_TEMPLATE, fewshot_prefix

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
# the final number is only trusted when the closing clause announces it, and never
# when the closing clause is a ratio ("Gold : Silver = 1 : 2" labels the row "2")
CUE = re.compile(r"(?i)(\bis\b|\bare\b|\bwas\b|\bwere\b|\bequals?\b|\bwill be\b|\bcosts?\b|\btotal(?:s|ling)?\b|\bapproximately\b|\bhas\b|\bhave\b|\bneeds?\b|\bgets?\b|\bmakes?\b|\bpaid\b|\bpays?\b|\bremain(?:s|ing)?\b|\bleft\b|=|:)\s*[^0-9]{0,24}$")
RATIO = re.compile(r"\d+\s*:\s*\d+")

def final_number(text):
    ms = list(NUM.finditer(text))
    if not ms:
        return None, None
    m = ms[-1]
    raw = m.group(0).replace(",", "").rstrip(".")
    if not re.fullmatch(r"-?\d+(\.\d+)?", raw):
        return None, None
    if raw.endswith(".0"):
        raw = raw[:-2]
    return raw, m.start()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/orca.jsonl")
    ap.add_argument("--n", type=int, default=95000)
    ap.add_argument("--fewshot-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    prefix = fewshot_prefix()
    d = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    idx = list(range(len(d))); rng.shuffle(idx)
    seen_q, out = set(), []
    stats = {"no_number": 0, "not_in_tail": 0, "too_long": 0, "marker": 0, "dup": 0,
             "no_cue": 0, "ratio": 0}
    for i in idx:
        if len(out) >= args.n:
            break
        r = d[i]
        q, a = r["question"].strip(), r["answer"].strip()
        if q in seen_q:
            stats["dup"] += 1; continue
        if "####" in a or "ANSWER:" in a:
            stats["marker"] += 1; continue
        if not (60 <= len(a) <= 2500) or len(q) > 1200:
            stats["too_long"] += 1; continue
        ans, pos = final_number(a)
        if ans is None:
            stats["no_number"] += 1; continue
        if pos < len(a) - 140:          # the answer must be in the closing sentence
            stats["not_in_tail"] += 1; continue
        if not CUE.search(a[:pos]):     # ...announced by the clause in front of it
            stats["no_cue"] += 1; continue
        if RATIO.search(a[max(0, pos - 60):]):
            stats["ratio"] += 1; continue
        seen_q.add(q)
        use_fs = rng.random() < args.fewshot_frac
        out.append({"id": f"orca-{len(out)}", "system": prefix if use_fs else None,
                    "user": MATH_PROMPT_TEMPLATE.format(prompt=q),
                    "target": f"{a}\n\nANSWER: {ans}<end_of_turn>",
                    "answer": ans, "src": "orca_math", "fewshot": bool(use_fs)})
    print("kept", len(out), "rejected", stats)
    with open(args.out, "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    with open(args.out.replace(".jsonl", "_forcheck.jsonl"), "w") as f:
        for r in out: f.write(json.dumps({"text": r["user"] + "\n" + r["target"]}) + "\n")
    print("wrote", args.out)

if __name__ == "__main__":
    main()
