#!/usr/bin/env python3
"""Build the SFT mixture.

Every row is normalised to one shape:
    reasoning ... \n\nANSWER: <number><end_of_turn>
so the grader's "last number in the completion" rule reads exactly one thing,
and generation stops where the grading template says a turn stops.

Sources (all TRAIN-side; the gsm8k test split is never touched):
  gsm8k       openai/gsm8k train (7473)                  human CoT
  omi2        nvidia/OpenMathInstruct-2 train_1M,        Llama-3.1-405B CoT on
              problem_source in {gsm8k, augmented_gsm8k} gsm8k-train seeds
  metamath    meta-math/MetaMathQA, GSM_* types          gsm8k-train rewrites
  orca        microsoft/orca-math-word-problems-200k     synthetic word problems
"""

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def norm_num(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUM_RE.fullmatch(s):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return ("%.10f" % f).rstrip("0").rstrip(".")


def strip_boxed(text: str) -> str:
    """Remove \\boxed{...} wrappers, keeping their contents."""
    out = []
    i = 0
    while i < len(text):
        j = text.find("\\boxed{", i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k = j + len("\\boxed{")
        depth = 1
        while k < len(text) and depth:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        out.append(text[j + len("\\boxed{") : k])
        i = k + 1
    return "".join(out)


def clean_body(text: str) -> str:
    text = strip_boxed(text)            # \boxed{x} -> x  (must run before any \boxed stripping)
    text = text.replace("\\boxed", "")  # leftovers from odd escapes
    text = re.sub(r"<<[^>]*>>", "", text)  # gsm8k calculator annotations
    # drop trailing answer declarations so "ANSWER: n" is the only final marker
    lines = [ln.rstrip() for ln in text.strip().split("\n")]
    while lines and re.match(
        r"^(the (final )?answer is[:\s]|####|\\\[|\$?\\?boxed)",
        lines[-1].strip().lower(),
    ):
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------- sources


def src_gsm8k():
    from datasets import load_dataset

    for r in load_dataset("openai/gsm8k", "main", split="train"):
        body, _, ans = r["answer"].rpartition("####")
        a = norm_num(ans)
        if a is None:
            continue
        yield {"question": r["question"], "body": clean_body(body), "answer": a,
               "source": "gsm8k"}


def src_omi2(max_rows: int):
    from datasets import load_dataset

    d = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    d = d.filter(
        lambda x: x["problem_source"] in ("gsm8k", "augmented_gsm8k"),
        num_proc=8,
    )
    print(f"  omi2 gsm8k-sourced rows: {len(d)}", flush=True)
    idx = list(range(len(d)))
    random.Random(0).shuffle(idx)
    n = 0
    for i in idx:
        if n >= max_rows:
            break
        r = d[i]
        a = norm_num(r["expected_answer"])
        if a is None:
            continue
        body = clean_body(r["generated_solution"])
        if not body:
            continue
        yield {"question": r["problem"], "body": body, "answer": a, "source": "omi2"}
        n += 1


def src_metamath(max_rows: int):
    from datasets import load_dataset

    d = load_dataset("meta-math/MetaMathQA", split="train")
    d = d.filter(lambda x: x["type"].startswith("GSM"), num_proc=8)
    print(f"  metamath GSM rows: {len(d)}", flush=True)
    idx = list(range(len(d)))
    random.Random(1).shuffle(idx)
    n = 0
    for i in idx:
        if n >= max_rows:
            break
        r = d[i]
        m = re.search(r"The answer is:?\s*(.+?)\s*$", r["response"], re.S)
        if not m:
            continue
        a = norm_num(strip_boxed(m.group(1)))
        if a is None:
            continue
        body = clean_body(r["response"][: m.start()])
        if not body:
            continue
        yield {"question": r["query"], "body": body, "answer": a, "source": "metamath"}
        n += 1


def src_orca(max_rows: int):
    from datasets import load_dataset

    d = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    idx = list(range(len(d)))
    random.Random(2).shuffle(idx)
    n = 0
    for i in idx:
        if n >= max_rows:
            break
        r = d[i]
        txt = strip_boxed(r["answer"])
        nums = NUM_RE.findall(txt)
        if not nums:
            continue
        a = norm_num(nums[-1])
        if a is None:
            continue
        body = clean_body(r["answer"])
        if not body or a not in body.replace(",", ""):
            continue
        yield {"question": r["question"], "body": body, "answer": a, "source": "orca"}
        n += 1


# ---------------------------------------------------------------- build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm8k", type=int, default=7473)
    ap.add_argument("--gsm8k-repeat", type=int, default=1)
    ap.add_argument("--n-omi2", type=int, default=0)
    ap.add_argument("--n-metamath", type=int, default=0)
    ap.add_argument("--n-orca", type=int, default=0)
    ap.add_argument("--fewshot-frac", type=float, default=0.25,
                    help="share of rows rendered with the grader's exact 10-shot system message")
    ap.add_argument("--max-target-tokens", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = fmt.get_tokenizer()
    system = fmt.fewshot_system_message()
    print(f"template sha {fmt.template_sha()}  fewshot chars {len(system)}", flush=True)

    rows = []
    if args.n_gsm8k:
        rows += list(src_gsm8k())[: args.n_gsm8k] * args.gsm8k_repeat
    if args.n_omi2:
        rows += list(src_omi2(args.n_omi2))
    if args.n_metamath:
        rows += list(src_metamath(args.n_metamath))
    if args.n_orca:
        rows += list(src_orca(args.n_orca))
    print(f"raw rows {len(rows)}", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    seen = set()
    kept, dropped_dup, dropped_len = [], 0, 0
    for r in rows:
        key = (re.sub(r"\s+", " ", r["question"]).strip().lower(), r["answer"])
        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)
        target = fmt.render_target(r["body"], r["answer"])
        if len(tok(target, add_special_tokens=False)["input_ids"]) > args.max_target_tokens:
            dropped_len += 1
            continue
        use_fs = rng.random() < args.fewshot_frac
        prompt = fmt.render_prompt(tok, r["question"], system if use_fs else None)
        kept.append({
            "question": r["question"],
            "answer": r["body"] + "\nANSWER: " + r["answer"],
            "prompt": prompt,
            "completion": target,
            "source": r["source"],
            "fewshot": int(use_fs),
        })

    print(f"kept {len(kept)}  dup {dropped_dup}  too_long {dropped_len}", flush=True)
    from collections import Counter
    print(Counter(r["source"] for r in kept), flush=True)

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
