#!/usr/bin/env python3
"""Build the SFT dataset for gemma-3-4b-pt -> GSM8K.

Target shape is dictated by the grader:
  * inspect_evals/gsm8k uses match(location="end", numeric=True) -> the LAST number
    in the completion must be the answer.
  * the harness renders with templates/gemma3.jinja -> assistant turns terminate with
    <end_of_turn>, and vLLM stops there (generation_config eos_token_id = [1, 106]).
So every target is:  <reasoning>\n\nANSWER: <int>   and the row ends with <end_of_turn>.

Sources (all GSM8K *train*-derived or independent; the test split is never read):
  nvidia/OpenMathInstruct-2   (problem_source in {gsm8k, augmented_gsm8k})
  openai/gsm8k                (train split, human CoT)
  microsoft/orca-math-word-problems-200k
"""
import argparse, json, os, random, re, sys
from collections import Counter

from datasets import load_dataset, load_from_disk

SNAP = os.environ["PTB_BASE_MODEL_SNAPSHOT"]
TASK = "/home/ben/task"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
INT_RE = re.compile(r"^-?\d+$")


def clean_int(s):
    """Return the canonical integer string, or None if the answer is not a plain int."""
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "").strip()
    s = s.rstrip(".")
    if INT_RE.match(s):
        return str(int(s))
    # allow x.0
    try:
        f = float(s)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except ValueError:
        pass
    return None


def strip_boxed(sol):
    """Remove \\boxed{...} wrappers, keeping their contents."""
    prev = None
    while prev != sol:
        prev = sol
        sol = BOXED.sub(r"\1", sol)
    return sol


def finalize(reasoning, ans):
    """reasoning + the single ANSWER line. Returns None if the body is unusable."""
    body = reasoning.strip()
    if not body:
        return None
    # the grader reads the last number; nothing may follow the ANSWER line
    if "ANSWER:" in body:
        return None
    return f"{body}\n\nANSWER: {ans}"


# ---------------------------------------------------------------- sources
def src_omi2(limit):
    d = load_from_disk(f"{TASK}/data/omi2_gsm8k_raw")
    rows = []
    for r in d:
        ans = clean_int(r["expected_answer"])
        if ans is None:
            continue
        sol = strip_boxed(r["generated_solution"])
        if "\\boxed" in sol or "####" in sol:
            continue
        t = finalize(sol, ans)
        if t is None:
            continue
        rows.append({"question": r["problem"].strip(), "target": t,
                     "answer": ans, "src": r["problem_source"]})
    random.Random(0).shuffle(rows)
    return rows[:limit]


def src_gsm8k_train(limit):
    d = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for r in d:
        body, _, tail = r["answer"].rpartition("####")
        ans = clean_int(tail)
        if ans is None:
            continue
        body = re.sub(r"<<[^>]*>>", "", body).strip()
        t = finalize(body, ans)
        if t is None:
            continue
        rows.append({"question": r["question"].strip(), "target": t,
                     "answer": ans, "src": "gsm8k_train"})
    random.Random(1).shuffle(rows)
    return rows[:limit]


ORCA_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def src_orca(limit):
    d = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    rows = []
    for r in d:
        a = r["answer"]
        if "\\boxed" in a or "####" in a or "ANSWER:" in a:
            continue
        ms = list(ORCA_NUM.finditer(a))
        if not ms:
            continue
        m = ms[-1]
        # the final number must be the payload, not a mid-sentence aside
        if len(a) - m.end() > 40:
            continue
        ans = clean_int(m.group(0))
        if ans is None:
            continue
        t = finalize(a, ans)
        if t is None:
            continue
        rows.append({"question": r["question"].strip(), "target": t,
                     "answer": ans, "src": "orca"})
    random.Random(2).shuffle(rows)
    return rows[:limit]


def src_metamath(limit):
    """GSM8K-train-derived rewrites. Bodies carry gsm8k's own '#### n' plus 'The answer is: n';
    both are cut so the row has exactly one answer marker (pitfall: double_answer_format)."""
    d = load_dataset("meta-math/MetaMathQA", split="train")
    rows = []
    for r in d:
        if r["type"] not in ("GSM_AnsAug", "GSM_Rephrased"):
            continue
        resp = r["response"]
        head, sep, tail = resp.partition("\n#### ")
        if not sep:
            continue
        ans = clean_int(tail.split("\n")[0])
        if ans is None or "####" in head or "answer is" in head.lower():
            continue
        t = finalize(head, ans)
        if t is None:
            continue
        rows.append({"question": r["query"].strip(), "target": t,
                     "answer": ans, "src": "metamath"})
    random.Random(4).shuffle(rows)
    return rows[:limit]


# ---------------------------------------------------------------- few-shot prefix
def fewshot_system_message():
    """Byte-identical to what inspect_evals/gsm8k builds for the eval prompt."""
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot
    fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                          sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
    return "\n\n".join(sample_to_fewshot(s) for s in fewshots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-omi2", type=int, default=75000)
    ap.add_argument("--n-gsm8k", type=int, default=7500)
    ap.add_argument("--n-orca", type=int, default=10000)
    ap.add_argument("--n-metamath", type=int, default=0)
    ap.add_argument("--n-fewshot", type=int, default=3000)
    ap.add_argument("--out", default=f"{TASK}/data/sft_v1.jsonl")
    args = ap.parse_args()

    rows = []
    rows += src_omi2(args.n_omi2)
    rows += src_gsm8k_train(args.n_gsm8k)
    rows += src_orca(args.n_orca)
    if args.n_metamath:
        rows += src_metamath(args.n_metamath)
    print(Counter(r["src"] for r in rows), file=sys.stderr)

    # dedup: identical (question, target) never twice; at most 2 solutions per question
    seen_pair, per_q, dedup = set(), Counter(), []
    for r in rows:
        k = re.sub(r"\s+", " ", r["question"].lower())
        pair = (k, r["target"])
        if pair in seen_pair or per_q[k] >= 2:
            continue
        seen_pair.add(pair)
        per_q[k] += 1
        dedup.append(r)
    print(f"dedup {len(rows)} -> {len(dedup)}", file=sys.stderr)
    rows = dedup

    sysmsg = fewshot_system_message()
    with open(f"{TASK}/data/fewshot_system.txt", "w") as f:
        f.write(sysmsg)

    rnd = random.Random(3)
    rnd.shuffle(rows)
    n_fs = min(args.n_fewshot, len(rows))
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            rec = {
                "system": sysmsg if i < n_fs else None,
                "user": MATH_PROMPT_TEMPLATE.format(prompt=r["question"]),
                # the stop token the grading template terminates on lives in the target itself
                "target": r["target"] + "<end_of_turn>",
                "answer": r["answer"],
                "src": r["src"],
                # 'text' is what the contamination checker reads
                "text": r["question"] + "\n\n" + r["target"],
            }
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} ({n_fs} with the 10-shot prefix)", file=sys.stderr)


if __name__ == "__main__":
    main()
