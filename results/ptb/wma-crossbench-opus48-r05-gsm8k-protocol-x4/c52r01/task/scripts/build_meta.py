#!/usr/bin/env python3
"""Build exp-02 augmented SFT data: GSM8K-train + MetaMathQA (GSM forward types).

MetaMathQA GSM responses carry TWO answer markers ("#### N" and "The answer is: N")
-> double_answer_format pitfall. We strip BOTH and emit exactly one "ANSWER: N".
Only GSM_AnsAug + GSM_Rephrased are used (forward word problems, numeric answers,
same distribution as the GSM8K test set). MATH_* excluded (non-numeric answers).
All source items derive from GSM8K/MATH TRAIN, never the test set; output is
contamination-checked against the test copy afterwards.
"""
import json, re, random

from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC = re.compile(r"<<[^>]*>>")
N_META = 48000
KEEP_TYPES = {"GSM_AnsAug", "GSM_Rephrased"}


def clean_num(s: str) -> str:
    s = s.strip().strip(".").replace("$", "").replace(",", "").strip()
    return s


def is_num(s: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", s))


def make_target(reasoning: str, ans: str):
    reasoning = CALC.sub("", reasoning)
    lines = []
    for ln in reasoning.split("\n"):
        if ln.strip().startswith("####"):
            continue
        if "the answer is" in ln.lower():
            continue
        lines.append(ln)
    reasoning = "\n".join(lines)
    reasoning = re.sub(r"[ \t]+", " ", reasoning)
    reasoning = re.sub(r" *\n *", "\n", reasoning).strip()
    if not reasoning:
        return None
    return f"{reasoning}\n\nANSWER: {ans}"


def emit(out_msgs, out_text, q, target_text):
    user = MATH_PROMPT_TEMPLATE.format(prompt=q.strip())
    out_msgs.write(json.dumps({
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": target_text},
        ],
        "completion": target_text + "<end_of_turn>",
    }) + "\n")
    out_text.write(json.dumps({"text": q.strip() + "\n" + target_text}) + "\n")


def main():
    random.seed(0)
    out_msgs = open("data/train_meta_messages.jsonl", "w")
    out_text = open("data/train_meta_text.jsonl", "w")
    n_gsm, n_meta, skipped = 0, 0, 0

    # 1) original GSM8K train (canonical, high quality)
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for r in gsm:
        parts = r["answer"].split("####")
        ans = clean_num(parts[-1])
        reasoning = "####".join(parts[:-1])
        t = make_target(reasoning, ans)
        if t and is_num(ans):
            emit(out_msgs, out_text, r["question"], t)
            n_gsm += 1

    # 2) MetaMathQA GSM forward subset
    meta = load_dataset("meta-math/MetaMathQA", split="train")
    idxs = [i for i, ty in enumerate(meta["type"]) if ty in KEEP_TYPES]
    random.shuffle(idxs)
    for i in idxs:
        if n_meta >= N_META:
            break
        r = meta[i]
        resp = r["response"]
        # extract answer: prefer '#### N', else 'The answer is: N'
        ans = None
        m = re.search(r"####\s*([^\n]+)", resp)
        if m:
            ans = clean_num(m.group(1))
        if not (ans and is_num(ans)):
            m2 = re.search(r"[Tt]he answer is:?\s*([^\n]+)", resp)
            if m2:
                ans = clean_num(m2.group(1))
        if not (ans and is_num(ans)):
            skipped += 1
            continue
        t = make_target(resp, ans)
        if not t:
            skipped += 1
            continue
        emit(out_msgs, out_text, r["query"], t)
        n_meta += 1

    out_msgs.close(); out_text.close()
    print(f"gsm={n_gsm} meta={n_meta} skipped={skipped} total={n_gsm+n_meta}")


if __name__ == "__main__":
    main()
