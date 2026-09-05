#!/usr/bin/env python3
"""RFT rejection sampling from a fine-tuned checkpoint on GSM8K-train questions.

Heeds vllm_offline_prompt_and_stop pitfall:
- render with the grader gemma template, tokenize add_special_tokens=False,
  pass prompt_token_ids (no double <bos>), assert not [2,2,...]
- explicit stop_token_ids=[1,106] for the n>1 route
- preserve raw draws; guard non-finite numeric parses as skipped
Keeps up to --keep correct, deduped solutions per question.
Output: data/rft.jsonl {prompt, completion} and data/rft_check.jsonl {question, answer}.
CPU/GPU model execution — runs only under a locked card.
"""
import os, re, json, argparse
os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"
STOP = "<end_of_turn>"
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

def norm_num(s):
    try:
        s = s.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
        v = float(s)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return round(v, 4)
    except Exception:
        return None

def gold_from_answer(ans):
    return norm_num(ans.rsplit("####", 1)[-1])

_LASTNUM = re.compile(r"-?\d[\d,]*\.?\d*")
def pred_from_text(text):
    # prefer the ANSWER: line, else last number in text
    m = re.findall(r"ANSWER:\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return norm_num(m[-1])
    nums = _LASTNUM.findall(text)
    return norm_num(nums[-1]) if nums else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="data/rft.jsonl")
    ap.add_argument("--out-check", default="data/rft_check.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.chat_template = open(TEMPLATE_PATH).read()

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    if args.limit:
        gsm = gsm.select(range(args.limit))

    questions = [ex["question"] for ex in gsm]
    golds = [gold_from_answer(ex["answer"]) for ex in gsm]

    rendered = [tok.apply_chat_template([{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q.strip())}],
                                        tokenize=False, add_generation_prompt=True) for q in questions]
    prompt_ids = [tok(r, add_special_tokens=False)["input_ids"] for r in rendered]
    # sanity: single bos, no double
    assert prompt_ids[0][0] == 2 and prompt_ids[0][1] != 2, prompt_ids[0][:4]
    print("prompt[0][:4]:", prompt_ids[0][:4], "n_prompts:", len(prompt_ids), flush=True)

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=2048, enforce_eager=False)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=0)

    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids]
    outs = llm.generate(prompts, sampling_params=sp)

    kept_rows, check_rows = [], []
    n_correct_any = 0
    finish_reasons = {}
    for i, out in enumerate(outs):
        gold = golds[i]
        seen = set()
        kept_here = 0
        any_ok = False
        for comp in out.outputs:
            fr = comp.finish_reason
            finish_reasons[fr] = finish_reasons.get(fr, 0) + 1
            text = comp.text.strip()
            if pred_from_text(text) is None or gold is None:
                continue
            if abs(pred_from_text(text) - gold) > 1e-4:
                continue
            any_ok = True
            # require clean single ANSWER marker
            if text.count("ANSWER:") != 1:
                continue
            key = text
            if key in seen:
                continue
            seen.add(key)
            if kept_here >= args.keep:
                continue
            completion = text + STOP
            kept_rows.append({"prompt": rendered[i], "completion": completion})
            check_rows.append({"question": questions[i], "answer": text})
            kept_here += 1
        if any_ok:
            n_correct_any += 1

    with open(args.out, "w") as f:
        for r in kept_rows:
            f.write(json.dumps(r) + "\n")
    with open(args.out_check, "w") as f:
        for r in check_rows:
            f.write(json.dumps(r) + "\n")
    print(f"questions: {len(questions)}  solvable(any correct): {n_correct_any} "
          f"({n_correct_any/len(questions):.3f})", flush=True)
    print(f"kept RFT rows: {len(kept_rows)}", flush=True)
    print("finish_reasons:", finish_reasons, flush=True)

if __name__ == "__main__":
    main()
