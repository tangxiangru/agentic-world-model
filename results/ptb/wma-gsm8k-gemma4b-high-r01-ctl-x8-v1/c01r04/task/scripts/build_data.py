#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt -> GSM8K.

Source: nvidia/OpenMathInstruct-2 (rev 469216e3f46f4dacf476b382e192485ea51a143e),
rows with problem_source in {gsm8k, augmented_gsm8k}.  Those are the original
GSM8K *train* problems and Llama-3.1-405B augmentations seeded from the *train*
split; the test split is never touched.

Output jsonl rows: {"prompt": <rendered user turn>, "completion": <target ending in <end_of_turn>>}
rendered byte-for-byte the way templates/gemma3.jinja + inspect_evals/gsm8k render
the graded prompt (see scripts/eval_format.py).
"""
import argparse, glob, json, random, re, sys, os
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_format import fewshot_prefix, fewshot_questions, render_prompt, render_target

OMI2 = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"

BOX = re.compile(r"\\boxed\s*\{([^{}]*)\}")
# non-negative only: the grader's match(numeric=True, location="end") reads the last
# whitespace-word for which word.replace(".","").isnumeric() is true, and "-7" fails that
# test, so a negative gold would silently be graded against an earlier number in the text.
NUMERIC = re.compile(r"^\d+(\.\d+)?$")


def clean_solution(sol: str) -> str | None:
    """Strip LaTeX boxing so the target carries exactly one answer marker."""
    if "\\boxed" not in sol:
        return None
    s = BOX.sub(r"\1", sol)
    if "\\boxed" in s:
        return None  # nested braces we did not unwrap
    s = s.replace("\\[", "").replace("\\]", "")
    s = s.replace("$$", "").replace("\\(", "").replace("\\)", "")
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    if not s:
        return None
    if "ANSWER:" in s or "####" in s:
        return None  # a second answer marker would confuse the grader
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/sft_train.jsonl")
    ap.add_argument("--dev-out", default="/home/ben/task/data/dev_gsm8ktrain.jsonl")
    ap.add_argument("--n-target", type=int, default=70000)
    ap.add_argument("--max-per-problem-orig", type=int, default=4)
    ap.add_argument("--max-per-problem-aug", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--max-sol-chars", type=int, default=2400)
    ap.add_argument("--dev-n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[], help="jsonl files whose (problem, solution) pairs are already used")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---- held-out internal dev set: GSM8K *train* problems we never train on ----
    from datasets import load_dataset
    tr = load_dataset("openai/gsm8k", "main", split="train")
    idx = list(range(len(tr)))
    random.Random(12345).shuffle(idx)
    dev_idx = idx[: args.dev_n]
    dev_problems = set()
    dev_rows = []
    for i in dev_idx:
        rec = tr[i]
        dev_problems.add(rec["question"].strip())
        dev_rows.append({
            "id": f"gsm8ktrain-{i}",
            "question": rec["question"].strip(),
            "gold": rec["answer"].split("####")[-1].strip(),
        })
    # the 10 eval few-shot problems are train items too; they are shown verbatim in
    # every graded prompt, so exclude them from training targets as well.
    fs_prefix = fewshot_prefix()
    dev_problems |= set(fewshot_questions())
    with open(args.dev_out, "w") as f:
        for r in dev_rows:
            f.write(json.dumps(r) + "\n")
    print(f"dev set: {len(dev_rows)} held-out gsm8k-train problems -> {args.dev_out}")

    # ---- load OpenMathInstruct-2 shards ----
    files = sorted(glob.glob(OMI2))
    assert files, "no OpenMathInstruct-2 parquet shards found"
    keep = []
    for fp in files:
        df = pq.read_table(fp, columns=["problem", "generated_solution",
                                        "expected_answer", "problem_source"]).to_pandas()
        df = df[df.problem_source.isin(["gsm8k", "augmented_gsm8k"])]
        keep.append(df)
        print(f"  {os.path.basename(fp)}: kept {len(df)}", flush=True)
    df = pd.concat(keep, ignore_index=True)
    print("pooled rows:", len(df))

    # ---- filter + cap solutions per problem ----
    probs = df["problem"].tolist()
    sols = df["generated_solution"].tolist()
    anss = df["expected_answer"].tolist()
    srcs = df["problem_source"].tolist()
    del df
    used = set()
    for xf in args.exclude:
        for line in open(xf):
            d = json.loads(line)
            used.add(hash((d["prompt"].rsplit("Solve the following math problem step by step.", 1)[-1],
                           d["completion"])))
    if args.exclude:
        print(f"excluding {len(used)} already-used (problem, solution) pairs")
    per_problem = {}
    rows = []
    order = list(range(len(probs)))
    rng.shuffle(order)
    for i in order:
        prob = probs[i].strip()
        if prob in dev_problems:
            continue                      # keep the internal dev set clean
        ans = str(anss[i]).strip()
        if not NUMERIC.match(ans):
            continue
        sol = sols[i]
        if not isinstance(sol, str) or len(sol) > args.max_sol_chars or len(sol) < 60:
            continue
        sol = clean_solution(sol)
        if sol is None:
            continue
        cap = args.max_per_problem_orig if srcs[i] == "gsm8k" else args.max_per_problem_aug
        k = per_problem.get(prob, 0)
        if k >= cap:
            continue
        if used:
            pr = render_prompt(prob, None).rsplit("Solve the following math problem step by step.", 1)[-1]
            if hash((pr, render_target(sol, ans))) in used:
                continue
        per_problem[prob] = k + 1
        rows.append({"problem": prob, "solution": sol, "answer": ans,
                     "src": srcs[i]})
        if len(rows) >= args.n_target:
            break
    print(f"selected {len(rows)} rows over {len(per_problem)} distinct problems")
    from collections import Counter
    print(Counter(r["src"] for r in rows))

    # ---- render to the grader's exact strings ----
    n_fs = int(len(rows) * args.fewshot_frac)
    fs_flag = [True] * n_fs + [False] * (len(rows) - n_fs)
    rng.shuffle(fs_flag)
    with open(args.out, "w") as f:
        for r, use_fs in zip(rows, fs_flag):
            prompt = render_prompt(r["problem"], fs_prefix if use_fs else None)
            completion = render_target(r["solution"], r["answer"])
            f.write(json.dumps({"prompt": prompt, "completion": completion,
                                "src": r["src"], "fewshot": use_fs}) + "\n")
    print(f"wrote {len(rows)} rows ({n_fs} with the 10-shot prefix) -> {args.out}")


if __name__ == "__main__":
    main()
