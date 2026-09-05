#!/usr/bin/env python3
"""Iteration 2 data: same CoT targets, but a fraction of examples get a
few-shot system context (matching the eval's structure) so the model learns to
answer ONE question and STOP (emit <end_of_turn>) even after several QA blocks.
"""
import json, re, random
random.seed(2)

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def clean_gsm8k(ans):
    parts = ans.split("####")
    reasoning = re.sub(r"<<[^>]*>>", "", parts[0]).strip()
    final = parts[1].strip().replace(",", "").replace("$", "") if len(parts) > 1 else ""
    return reasoning, final


def norm_num(s):
    return s.strip().replace(",", "").replace("$", "").rstrip(".")


def main():
    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="train")

    # raw fewshot pool in the EXACT eval format (sample_to_fewshot: raw reasoning w/ <<>>)
    fewshot_pool = []
    for r in gsm:
        reasoning_raw = r["answer"].split("####")[0].strip()
        _, final = clean_gsm8k(r["answer"])
        if not final:
            continue
        block = f"{r['question'].strip()}\n\nReasoning:\n{reasoning_raw}\n\nANSWER: {final}"
        fewshot_pool.append(block)
    print("fewshot pool:", len(fewshot_pool))

    examples = []
    for r in gsm:
        reasoning, final = clean_gsm8k(r["answer"])
        if not final:
            continue
        examples.append({"question": r["question"].strip(),
                         "reasoning_answer": f"{reasoning}\nANSWER: {final}"})

    mm = load_dataset("meta-math/MetaMathQA", split="train")
    gsm_types = {"GSM_AnsAug", "GSM_Rephrased", "GSM_FOBAR", "GSM_SV"}
    ans_re = re.compile(r"The answer is:\s*(.*?)\s*$", re.DOTALL)
    mm_examples = []
    for r in mm:
        if r["type"] not in gsm_types:
            continue
        m = ans_re.search(r["response"].strip())
        if not m:
            continue
        final = norm_num(m.group(1).split("\n")[0])
        if not re.fullmatch(r"-?\d+(\.\d+)?", final):
            continue
        reasoning = re.sub(r"\\boxed\{([^}]*)\}", r"\1", r["response"][:m.start()].strip())
        if len(reasoning) < 10:
            continue
        mm_examples.append({"question": r["query"].strip(),
                            "reasoning_answer": f"{reasoning}\nANSWER: {final}"})
    random.shuffle(mm_examples)
    mm_examples = mm_examples[:60000]
    examples.extend(mm_examples)
    random.shuffle(examples)
    print("total base examples:", len(examples))

    # Assign few-shot system context to ~40% of examples.
    FEWSHOT_FRAC = 0.30
    for e in examples:
        if random.random() < FEWSHOT_FRAC:
            k = random.randint(3, 8)
            shots = random.sample(fewshot_pool, k)
            # avoid a shot identical to the target question
            shots = [s for s in shots if not s.startswith(e["question"][:60])][:k]
            e["system"] = "\n\n".join(shots)

    with open("train_data3.jsonl", "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    nfs = sum(1 for e in examples if "system" in e)
    print("with fewshot system:", nfs, "/", len(examples))


if __name__ == "__main__":
    main()
