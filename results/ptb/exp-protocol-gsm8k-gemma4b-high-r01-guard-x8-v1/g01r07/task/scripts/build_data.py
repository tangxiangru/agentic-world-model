#!/usr/bin/env python3
"""Build the SFT corpus.

Every row is rendered with the grader's own chat template (templates/gemma3.jinja,
hash-checked) so training and grading see byte-identical strings, and every target
ends with the terminator the grader stops on.

Output: data/sft_train.jsonl with {prompt, completion, src, kshot}
        data/sft_check.jsonl with {text} for the contamination checker
"""
from __future__ import annotations
import argparse, hashlib, json, random, re, sys
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TPL_PATH = "templates/gemma3.jinja"
STOP = "<end_of_turn>"

# byte-for-byte from inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def clean_number(s: str) -> str | None:
    """Return the canonical numeric string, or None if the answer is not a plain number."""
    s = str(s).strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUM_RE.match(s):
        return None
    if "." in s:
        s = s.rstrip("0").rstrip(".")
        if s in ("", "-"):
            return None
    try:
        float(s)
    except ValueError:
        return None
    return s


def strip_calc(txt: str) -> str:
    """Drop GSM8K's <<48/2=24>> calculator annotations."""
    return re.sub(r"<<[^>]*>>", "", txt)


def finalize(body: str, answer: str) -> str:
    """One body, one answer marker, terminator last."""
    body = body.strip()
    # the grader reads the LAST number in the completion; ANSWER must be the last line
    return f"{body}\n\nANSWER: {answer}"


# ---------------------------------------------------------------- sources
def src_gsm8k_train():
    d = load_dataset("openai/gsm8k", "main", split="train")
    for r in d:
        ans = clean_number(r["answer"].split("####")[-1])
        if ans is None:
            continue
        body = strip_calc("####".join(r["answer"].split("####")[:-1])).strip()
        if not body:
            continue
        yield r["question"].strip(), finalize(body, ans), "gsm8k_train"


def src_metamath(n_ansaug, n_rephrased, n_sv, n_fobar, rng):
    d = load_dataset("meta-math/MetaMathQA", split="train")
    want = {"GSM_AnsAug": n_ansaug, "GSM_Rephrased": n_rephrased,
            "GSM_SV": n_sv, "GSM_FOBAR": n_fobar}
    pools = {k: [] for k in want}
    for r in d:
        t = r["type"]
        if t in pools:
            pools[t].append(r)
    for t, k in want.items():
        pool = pools[t]
        rng.shuffle(pool)
        taken = 0
        for r in pool:
            if taken >= k:
                break
            resp = r["response"]
            # MetaMath carries TWO answer markers: "#### N" and "The answer is: N".
            # Both must go, or the model learns a second marker (pitfall double_answer_format).
            m = re.search(r"The answer is:\s*(.+?)\s*$", resp)
            if not m:
                continue
            ans = clean_number(m.group(1))
            if ans is None:
                continue
            body = resp[: m.start()]
            body = re.sub(r"####\s*[^\n]*\n?", "", body)
            body = strip_calc(body).strip()
            if not body:
                continue
            yield r["query"].strip(), finalize(body, ans), "metamath_" + t
            taken += 1


def src_omi2(path, sources, n_per, rng, max_chars=2600):
    pools = {s: [] for s in sources}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            s = r["problem_source"]
            if s in pools:
                pools[s].append(r)
    for s, k in zip(sources, n_per):
        pool = pools[s]
        rng.shuffle(pool)
        taken = 0
        for r in pool:
            if taken >= k:
                break
            ans = clean_number(r["expected_answer"])
            if ans is None:
                continue
            sol = r["generated_solution"]
            if len(sol) > max_chars:
                continue
            # unwrap \boxed{...} so the body carries no second answer marker
            body = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", sol)
            if "\\boxed" in body:
                continue
            body = body.strip()
            if not body:
                continue
            yield r["problem"].strip(), finalize(body, ans), "omi2_" + s
            taken += 1


def src_omi2_math(path, n, rng, max_chars=2600):
    yield from src_omi2(path, ["math", "augmented_math"], [n // 3, n - n // 3], rng, max_chars)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_train.jsonl")
    ap.add_argument("--check-out", default="data/sft_check.jsonl")
    ap.add_argument("--kshot-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-aug-gsm", type=int, default=40000)
    ap.add_argument("--n-omi-gsm", type=int, default=14764)
    ap.add_argument("--n-mm-ansaug", type=int, default=8000)
    ap.add_argument("--n-mm-rephrased", type=int, default=8000)
    ap.add_argument("--n-mm-sv", type=int, default=2000)
    ap.add_argument("--n-mm-fobar", type=int, default=2000)
    ap.add_argument("--n-math", type=int, default=6000)
    ap.add_argument("--max-per-question", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    tpl = open(TPL_PATH).read()
    print("template sha256:", hashlib.sha256(tpl.encode()).hexdigest())
    tok = AutoTokenizer.from_pretrained(SNAP)
    tok.chat_template = tpl
    sysmsg = open("data/eval_system_message.txt").read()

    rows = []
    gen = []
    gen += list(src_gsm8k_train())
    gen += list(src_omi2("data/omi2_gsm.jsonl", ["gsm8k", "augmented_gsm8k"],
                         [args.n_omi_gsm, args.n_aug_gsm], rng))
    gen += list(src_metamath(args.n_mm_ansaug, args.n_mm_rephrased,
                             args.n_mm_sv, args.n_mm_fobar, rng))
    gen += list(src_omi2_math("data/omi2_gsm_math.jsonl", args.n_math, rng))

    # dedup: drop identical (question, solution) pairs, and cap distinct solutions
    # per question so one problem cannot dominate the corpus
    seen, per_q = set(), Counter()
    uniq = []
    for q, c, s in gen:
        qk = q.lower()[:400]
        key = (qk, re.sub(r"\s+", " ", c.lower())[:200])
        if key in seen or per_q[qk] >= args.max_per_question:
            continue
        seen.add(key)
        per_q[qk] += 1
        uniq.append((q, c, s))
    print(f"{len(gen)} generated -> {len(uniq)} after dedup")
    rng.shuffle(uniq)

    n_kshot = int(len(uniq) * args.kshot_frac)
    for i, (q, c, s) in enumerate(uniq):
        user = MATH_PROMPT_TEMPLATE.replace("{prompt}", q)
        kshot = i < n_kshot
        msgs = ([{"role": "system", "content": sysmsg}] if kshot else []) + \
               [{"role": "user", "content": user}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        rows.append({"prompt": prompt, "completion": c + STOP, "src": s,
                     "kshot": int(kshot), "question": q})

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(args.check_out, "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["question"] + "\n" + r["completion"]}) + "\n")

    print(Counter(r["src"] for r in rows))
    print("kshot rows:", sum(r["kshot"] for r in rows), "of", len(rows))

    # ---- length audit (pitfall seq_len_truncation) ----
    samp = rows if len(rows) <= 4000 else rng.sample(rows, 4000)
    lens, clens = [], []
    for r in samp:
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c = tok(r["completion"], add_special_tokens=False)["input_ids"]
        lens.append(len(p) + len(c))
        clens.append(len(c))
    lens.sort()
    print("total len  p50/p95/p99/max:", lens[len(lens) // 2], lens[int(.95 * len(lens))],
          lens[int(.99 * len(lens))], lens[-1])
    clens.sort()
    print("completion p50/p95/max:", clens[len(clens) // 2], clens[int(.95 * len(clens))], clens[-1])
    print("mean total len:", sum(lens) / len(lens))

    # ---- verify the target really ends with the terminator ----
    bad = [r for r in rows[:2000] if not r["completion"].endswith(STOP)]
    print("targets not ending in stop token (first 2000):", len(bad))
    ids = tok(rows[0]["completion"], add_special_tokens=False)["input_ids"]
    print("last 3 target token ids:", ids[-3:], "->", tok.convert_ids_to_tokens(ids[-3:]))
    print("\n===== EXAMPLE ROW (kshot=0) =====")
    ex = next(r for r in rows if not r["kshot"])
    print(ex["prompt"][-700:])
    print("---COMPLETION---")
    print(ex["completion"])


if __name__ == "__main__":
    main()
