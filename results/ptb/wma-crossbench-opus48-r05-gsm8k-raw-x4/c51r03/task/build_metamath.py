import re, json, random
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

random.seed(0)
GSM_TYPES = {"GSM_Rephrased", "GSM_AnsAug", "GSM_SV", "GSM_FOBAR"}

def extract_final(resp):
    m = re.search(r"The answer is:\s*(.+?)\s*$", resp.strip(), re.DOTALL)
    if not m:
        return None, None
    ans = m.group(1).strip()
    # strip everything after "The answer is:" line from reasoning
    reasoning = resp[:m.start()].strip()
    # remove the gsm8k-style "#### N" answer line(s) to avoid double-answer pattern
    reasoning = re.sub(r"####.*", "", reasoning).strip()
    # remove \boxed{...}
    reasoning = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", reasoning)
    # remove leftover calculator annotations if any
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning)
    ans = ans.replace(",", "").replace("$", "").strip()
    # accept only clean numeric answers (int or decimal)
    if not re.fullmatch(r"-?\d+(\.\d+)?", ans):
        return None, None
    # normalize x.0 -> x
    if re.fullmatch(r"-?\d+\.0+", ans):
        ans = str(int(float(ans)))
    return reasoning, ans

def main():
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    fsft = open("data/metamath_gsm_sft.jsonl", "w")
    fdecon = open("data/metamath_gsm_decon.jsonl", "w")
    n = 0; skip = 0
    for r in ds:
        if r["type"] not in GSM_TYPES:
            continue
        reasoning, ans = extract_final(r["response"])
        if ans is None or not reasoning:
            skip += 1; continue
        q = r["query"].strip()
        assistant = f"{reasoning}\n\nANSWER: {ans}"
        user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        fsft.write(json.dumps({"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]}) + "\n")
        fdecon.write(json.dumps({"question": q, "answer": assistant}) + "\n")
        n += 1
    fsft.close(); fdecon.close()
    print("wrote", n, "skipped", skip)

if __name__ == "__main__":
    main()
