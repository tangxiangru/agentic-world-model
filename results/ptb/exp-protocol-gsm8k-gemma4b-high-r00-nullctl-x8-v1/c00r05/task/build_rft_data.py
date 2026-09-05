#!/usr/bin/env python3
"""Turn raw rejection-sampling output into a stage-2 SFT file."""
import argparse, json, random, re, collections, glob
import pandas as pd

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BAD = re.compile(r"(typo|mistake in the problem|incorrect solution|wait,|i apologize|"
                 r"this is wrong|re-evaluate|let me reconsider|problem statement is|"
                 r"seems to be flawed|does not make sense)", re.I)


def sig(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return tuple(n.replace(",", "") for n in nums)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/rft_raw.jsonl")
    ap.add_argument("--out", default="data/rft_sft.jsonl")
    ap.add_argument("--max-gsm", type=int, default=4)
    ap.add_argument("--max-aug", type=int, default=1)
    ap.add_argument("--fewshot-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    fs_pool, gsm_questions = [], []
    for r in gsm:
        parts = r["answer"].split("####")
        fs_pool.append((r["question"], "####".join(parts[:-1]).strip(),
                        parts[-1].strip().replace(",", "")))
        gsm_questions.append(r["question"])

    by_q = collections.defaultdict(list)
    src_of = {}
    attempted = collections.Counter()
    n_raw = n_ok = 0
    with open(args.raw) as f:
        for line in f:
            d = json.loads(line)
            n_raw += 1
            attempted[d["qid"]] += 1
            src_of[d["qid"]] = d["source"]
            if not d["correct"]:
                continue
            g = d["gen"].strip()
            if BAD.search(g) or len(g) < 30 or "ANSWER:" not in g:
                continue
            idx = g.find("ANSWER:")
            tail = g[idx:].split("\n")[0].strip()
            g = (g[:idx].rstrip() + "\n\n" + tail).strip()
            n_ok += 1
            by_q[d["qid"]].append((d["problem"], g))

    print(f"raw={n_raw} usable={n_ok} solved_problems={len(by_q)} / {len(attempted)}")

    rows = []
    for qid, lst in by_q.items():
        cap = args.max_gsm if src_of[qid] == "gsm8k_train" else args.max_aug
        seen, kept = set(), []
        rng.shuffle(lst)
        lst.sort(key=lambda x: len(x[1]))
        for prob, g in lst:
            s = sig(g)
            if s in seen:
                continue
            seen.add(s)
            kept.append((prob, g))
            if len(kept) >= cap:
                break
        rows.extend(kept)
    print("rft rows", len(rows))

    # supplement: off-policy OMI-2 solutions for GSM8K train problems never solved
    unsolved = {gsm_questions[int(q[1:])] for q in attempted
                if src_of[q] == "gsm8k_train" and q not in by_q}
    print("unsolved gsm8k train problems:", len(unsolved))
    if unsolved:
        from prep_data import clean_solution, is_int_answer, norm_int
        got = collections.Counter()
        files = sorted(glob.glob(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"))
        extra = 0
        for f in files:
            df = pd.read_parquet(f, columns=["problem", "generated_solution",
                                             "expected_answer", "problem_source"])
            for prob, sol, ans, src in df.itertuples(index=False):
                if src != "gsm8k" or prob not in unsolved or not is_int_answer(ans):
                    continue
                if got[prob] >= 4 or len(sol) > 3000:
                    continue
                got[prob] += 1
                rows.append((prob, f"{clean_solution(sol)}\n\nANSWER: {norm_int(ans)}"))
                extra += 1
        print("supplement rows", extra, "covering", len(got))

    rng.shuffle(rows)
    with open(args.out, "w") as fo:
        for prob, g in rows:
            user = MATH_PROMPT_TEMPLATE.format(prompt=prob.strip())
            if rng.random() < args.fewshot_frac:
                shots = rng.sample(fs_pool, rng.randint(1, 10))
                block = "\n\n".join(f"{a}\n\nReasoning:\n{b}\n\nANSWER: {c}" for a, b, c in shots)
                user = block + "\n\n" + user
            fo.write(json.dumps({"prompt": user, "response": g, "problem": prob}) + "\n")
    print("wrote", args.out, len(rows))


if __name__ == "__main__":
    main()
