#!/usr/bin/env python3
"""Format MetaMathQA GSM subset into eval-matching SFT data (numeric answers only)."""
import json
import re

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def extract_answer(response: str):
    m = re.search(r"The answer is:\s*(.*)", response, re.DOTALL)
    if not m:
        return None, None
    ans = m.group(1).strip()
    # take first line of the answer
    ans = ans.split("\n")[0].strip()
    ans = ans.replace("$", "").replace("\\boxed{", "").replace("}", "").strip()
    ans = ans.rstrip(".")
    # numeric only: remove commas
    a = ans.replace(",", "").strip()
    # keep only if it's a clean number (int or decimal, optional sign)
    if not re.fullmatch(r"-?\d+(\.\d+)?", a):
        return None, None
    # reasoning = everything before "The answer is:"
    reasoning = response[: m.start()].strip()
    # strip trailing boxed sentence remnants and calc annotations
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning)
    # remove any "#### X" gsm-style answer lines left in the reasoning
    reasoning = re.sub(r"\n?####.*", "", reasoning).strip()
    return reasoning, a


def main():
    kept = 0
    total = 0
    with open("metamath_raw.jsonl") as f, open("metamath_gsm.jsonl", "w") as out:
        for line in f:
            total += 1
            r = json.loads(line)
            if not r["type"].startswith("GSM"):
                continue
            reasoning, ans = extract_answer(r["response"])
            if reasoning is None or len(reasoning) < 10:
                continue
            q = r["query"].strip()
            completion = f"{reasoning}\n\nANSWER: {ans}"
            user = MATH_PROMPT_TEMPLATE.format(prompt=q)
            out.write(json.dumps({
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": completion},
                ],
                "question": q,
                "target": ans,
            }) + "\n")
            kept += 1
    print(f"total {total}, kept GSM numeric {kept}")


if __name__ == "__main__":
    main()
