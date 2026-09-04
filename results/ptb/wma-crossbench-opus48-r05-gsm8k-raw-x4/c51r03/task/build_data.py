import re, json
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

def clean_reasoning(ans: str):
    # split off final answer
    parts = ans.split("####")
    reasoning = parts[0].strip()
    final = parts[1].strip() if len(parts) > 1 else ""
    # remove calculator annotations <<...>>
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning)
    # normalize final number (strip commas, $)
    final_clean = final.replace(",", "").replace("$", "").strip()
    return reasoning, final_clean

def main():
    ds = load_dataset("openai/gsm8k", "main")["train"]
    # for contamination check (raw q + full answer text)
    fdecon = open("data/gsm8k_train_decon.jsonl", "w")
    # for SFT (chat messages)
    fsft = open("data/gsm8k_sft.jsonl", "w")
    n = 0
    for r in ds:
        q = r["question"].strip()
        reasoning, final = clean_reasoning(r["answer"])
        if not final:
            continue
        assistant = f"{reasoning}\n\nANSWER: {final}"
        user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        fsft.write(json.dumps({"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]}) + "\n")
        fdecon.write(json.dumps({"question": q, "answer": assistant}) + "\n")
        n += 1
    fsft.close(); fdecon.close()
    print("wrote", n, "examples")

if __name__ == "__main__":
    main()
