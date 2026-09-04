#!/usr/bin/env python3
"""exp-04 data: few-shot-CONTEXT SFT blending GSM8K-train + MetaMathQA GSM subset.

exp-03 fixed stopping but trained only on GSM8K-train reasoning (same coverage as
exp-01). This adds MetaMathQA GSM rephrasings/answer-augmentations (alternate correct
solution paths + phrasings of grade-school problems) AS few-shot-context rows so the
stopping fix is preserved: every row optionally carries k GSM8K-train exemplars in the
grader's EXACT format as a system prefix, then the target question, then a clean
completion ending in one 'ANSWER: N' + <end_of_turn>.

Exemplars in the prefix are ALWAYS GSM8K-train raw (matches the grader's exemplar
style). Targets come from GSM8K-train and MetaMathQA GSM forward types. All source
items derive from TRAIN splits; output is contamination-checked against the test copy.
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
K_CHOICES = [0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 8]
N_META = 16000
KEEP_TYPES = {"GSM_AnsAug", "GSM_Rephrased"}


def clean_num(s):
    return s.strip().strip(".").replace("$", "").replace(",", "").strip()


def is_num(s):
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", s))


def clean_gsm(ans):
    parts = ans.split("####")
    target = clean_num(parts[-1])
    reasoning = "####".join(parts[:-1]).strip()
    reasoning = CALC.sub("", reasoning)
    reasoning = re.sub(r"[ \t]+", " ", reasoning)
    reasoning = re.sub(r" *\n *", "\n", reasoning).strip()
    return reasoning, target


def clean_meta(resp, ans):
    resp = CALC.sub("", resp)
    lines = []
    for ln in resp.split("\n"):
        if ln.strip().startswith("####"):
            continue
        if "the answer is" in ln.lower():
            continue
        lines.append(ln)
    reasoning = "\n".join(lines)
    reasoning = re.sub(r"[ \t]+", " ", reasoning)
    reasoning = re.sub(r" *\n *", "\n", reasoning).strip()
    return reasoning


def raw_fewshot(q, raw_answer):
    a = raw_answer.split("####")
    target = a.pop().strip()
    reasoning = "####".join(a).strip()
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def main():
    random.seed(0)
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    # exemplar pool + GSM targets (both from raw GSM train)
    pool_raw, targets = [], []
    for r in gsm:
        reasoning, target = clean_gsm(r["answer"])
        if reasoning and is_num(target):
            q = r["question"].strip()
            pool_raw.append((q, r["answer"]))
            targets.append({"q": q, "reasoning": reasoning, "target": target})
    n_gsm = len(targets)

    # MetaMath GSM forward targets
    meta = load_dataset("meta-math/MetaMathQA", split="train")
    idxs = [i for i, ty in enumerate(meta["type"]) if ty in KEEP_TYPES]
    random.shuffle(idxs)
    n_meta = 0
    for i in idxs:
        if n_meta >= N_META:
            break
        r = meta[i]
        resp = r["response"]
        ans = None
        m = re.search(r"####\s*([^\n]+)", resp)
        if m:
            ans = clean_num(m.group(1))
        if not (ans and is_num(ans)):
            m2 = re.search(r"[Tt]he answer is:?\s*([^\n]+)", resp)
            if m2:
                ans = clean_num(m2.group(1))
        if not (ans and is_num(ans)):
            continue
        reasoning = clean_meta(resp, ans)
        if not reasoning:
            continue
        targets.append({"q": r["query"].strip(), "reasoning": reasoning, "target": ans})
        n_meta += 1

    random.shuffle(targets)
    P = len(pool_raw)
    out_msgs = open("data/train_fs2_messages.jsonl", "w")
    out_text = open("data/train_fs2_text.jsonl", "w")
    written = 0
    for it in targets:
        k = random.choice(K_CHOICES)
        msgs = []
        if k > 0:
            js = random.sample(range(P), k)
            shots = [raw_fewshot(pool_raw[j][0], pool_raw[j][1]) for j in js]
            msgs.append({"role": "system", "content": "\n\n".join(shots)})
        assistant = f'{it["reasoning"]}\n\nANSWER: {it["target"]}'
        msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=it["q"])})
        msgs.append({"role": "assistant", "content": assistant})
        out_msgs.write(json.dumps({"messages": msgs, "completion": assistant + "<end_of_turn>"}) + "\n")
        out_text.write(json.dumps({"text": it["q"] + "\n" + assistant}) + "\n")
        written += 1
    out_msgs.close(); out_text.close()
    print(f"wrote {written} rows: gsm={n_gsm} meta={n_meta}")


if __name__ == "__main__":
    main()
