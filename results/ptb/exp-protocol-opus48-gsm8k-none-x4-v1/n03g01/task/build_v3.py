#!/usr/bin/env python3
"""Build few-shot-augmented SFT data matching the eval's input format.

Prepends k random GSM8K-train exemplars (formatted exactly like the eval's
few-shot, including <<calc>> annotations) so the model learns to produce
exactly ONE answer and then STOP, even with multi-example context.
"""
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
    reasoning_raw = "####".join(parts[:-1]).strip()  # keeps <<...>>
    reasoning_clean = re.sub(r"<<[^>]*>>", "", reasoning_raw)
    return reasoning_raw, reasoning_clean, final


def fewshot_block(q, reasoning_raw, final):
    # matches inspect's sample_to_fewshot exactly
    return f"{q}\n\nReasoning:\n{reasoning_raw}\n\nANSWER: {final}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft", default="rft_full.jsonl")
    ap.add_argument("--out", default="train_v3.jsonl")
    ap.add_argument("--keep_rft", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    ds = load_dataset("openai/gsm8k", "main")["train"]
    exemplars = []   # (q, reasoning_raw, final) pool for few-shot prefixes
    gold_by_q = {}
    questions = []
    for r in ds:
        q = r["question"].strip()
        raw, clean, final = gold_parts(r["answer"])
        exemplars.append((q, raw, final))
        gold_by_q[q] = (clean, final)
        questions.append(q)

    # load RFT correct solutions
    rft_by_q = defaultdict(list)
    try:
        for l in open(args.rft):
            d = json.loads(l)
            rft_by_q[d["question"]].append(d["completion"])
    except FileNotFoundError:
        pass

    n = len(exemplars)
    out = []

    def make_example(q, completion):
        # choose k few-shot exemplars (not equal to q)
        k = random.randint(0, args.kmax)
        prompt_user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        if k > 0:
            picks = []
            while len(picks) < k:
                idx = random.randrange(n)
                if exemplars[idx][0] != q:
                    picks.append(exemplars[idx])
            sys_content = "\n\n".join(fewshot_block(*p) for p in picks)
            prompt_field = sys_content + "\n\n" + prompt_user
        else:
            prompt_field = prompt_user
        out.append({"prompt": prompt_field, "completion": completion})

    for q in questions:
        clean, final = gold_by_q[q]
        gold_completion = f"{clean}\n\nANSWER: {final}"
        make_example(q, gold_completion)
        for comp in rft_by_q.get(q, [])[:args.keep_rft]:
            make_example(q, comp)

    random.shuffle(out)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} examples (gold={len(questions)}, "
          f"rft_added={len(out)-len(questions)})")


if __name__ == "__main__":
    main()
