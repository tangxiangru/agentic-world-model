#!/usr/bin/env python3
"""Build v4: gold + merged RFT (from multiple rounds), few-shot augmented."""
import json, re, random, argparse
from datasets import load_dataset
from collections import defaultdict

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def gold_parts(answer):
    parts = answer.split("####")
    final = parts[-1].strip().replace(",", "")
    raw = "####".join(parts[:-1]).strip()
    clean = re.sub(r"<<[^>]*>>", "", raw)
    return raw, clean, final


def fewshot_block(q, raw, final):
    return f"{q}\n\nReasoning:\n{raw}\n\nANSWER: {final}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", nargs="+", default=["rft_full.jsonl", "rft_v3.jsonl"])
    ap.add_argument("--out", default="train_v4.jsonl")
    ap.add_argument("--keep_rft", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    ds = load_dataset("openai/gsm8k", "main")["train"]
    exemplars, gold_by_q, questions = [], {}, []
    for r in ds:
        q = r["question"].strip()
        raw, clean, final = gold_parts(r["answer"])
        exemplars.append((q, raw, final))
        gold_by_q[q] = (clean, final)
        questions.append(q)

    # merge RFT sources, dedup by normalized reasoning per question
    rft_by_q = defaultdict(list)
    seen_by_q = defaultdict(set)
    for path in args.rft:
        try:
            for l in open(path):
                d = json.loads(l)
                q = d["question"]; c = d["completion"]
                key = re.sub(r"\s+", " ", c)[:200]
                if key in seen_by_q[q]:
                    continue
                seen_by_q[q].add(key)
                rft_by_q[q].append(c)
        except FileNotFoundError:
            print("missing", path)

    n = len(exemplars)
    out = []

    def make(q, completion):
        k = random.randint(0, args.kmax)
        user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        if k > 0:
            picks = []
            while len(picks) < k:
                e = exemplars[random.randrange(n)]
                if e[0] != q:
                    picks.append(e)
            user = "\n\n".join(fewshot_block(*p) for p in picks) + "\n\n" + user
        out.append({"prompt": user, "completion": completion})

    n_solved = 0
    for q in questions:
        clean, final = gold_by_q[q]
        make(q, f"{clean}\n\nANSWER: {final}")
        rlist = rft_by_q.get(q, [])
        if rlist:
            n_solved += 1
        # shuffle then keep to diversify across rounds
        random.shuffle(rlist)
        for comp in rlist[:args.keep_rft]:
            make(q, comp)

    random.shuffle(out)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"total={len(out)} gold={len(questions)} solved_by_rft={n_solved} "
          f"rft_added={len(out)-len(questions)}")


if __name__ == "__main__":
    main()
