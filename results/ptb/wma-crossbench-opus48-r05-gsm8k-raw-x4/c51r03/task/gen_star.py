import re, json, argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

def gold_answer(ans):
    return ans.split("####")[-1].strip().replace(",", "").replace("$", "")

def extract_pred(text):
    # find last number after ANSWER: or last number overall
    m = re.findall(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None

def norm(x):
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except:
        return str(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/star_sft.jsonl")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--keep", type=int, default=3, help="max correct kept per question")
    ap.add_argument("--maxq", type=int, default=100000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open("templates/gemma3.jinja").read()

    ds = load_dataset("openai/gsm8k", "main")["train"]
    questions, golds, prompts = [], [], []
    for r in list(ds)[:args.maxq]:
        q = r["question"].strip()
        user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        p = tok.apply_chat_template([{"role": "user", "content": user}],
                                    add_generation_prompt=True, tokenize=False)
        questions.append(q); golds.append(norm(gold_answer(r["answer"]))); prompts.append(p)

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=2048, enable_prefix_caching=True)
    sp = SamplingParams(n=args.n, temperature=args.temp, top_p=0.95, max_tokens=768,
                        stop=["<end_of_turn>"])
    outs = llm.generate(prompts, sp)

    fout = open(args.out, "w")
    n_written = 0; n_solved = 0
    for q, gold, out in zip(questions, golds, outs):
        kept = []
        seen = set()
        for o in out.outputs:
            text = o.text.strip()
            pred = extract_pred(text)
            if pred is None:
                continue
            if norm(pred) == gold:
                # ensure it ends cleanly with ANSWER line
                key = text[:80]
                if key in seen:
                    continue
                seen.add(key)
                # normalize ending to ANSWER: gold
                if "ANSWER:" not in text:
                    text = text + f"\n\nANSWER: {gold}"
                kept.append(text)
            if len(kept) >= args.keep:
                break
        if kept:
            n_solved += 1
        user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        for text in kept:
            fout.write(json.dumps({"messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": text},
            ]}) + "\n")
            n_written += 1
    fout.close()
    print(f"questions {len(questions)} solved {n_solved} ({100*n_solved/len(questions):.1f}%) written {n_written}")

if __name__ == "__main__":
    main()
