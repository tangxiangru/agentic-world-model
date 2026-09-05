#!/usr/bin/env python3
"""RFT / rejection-sampling: sample K solutions per GSM8K-train question from an
SFT model, keep only those whose 'ANSWER: N' equals the gold answer, dedup.
Follows the vllm offline pitfall guidance: prompts are rendered with the grader
template then tokenized with add_special_tokens=False (the template already emits
<bos>), and explicit stop_token_ids=[<eos>,<end_of_turn>] are set."""
import argparse
import json
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,\.]+)")


def norm(s):
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", default="templates/gemma3.jinja")
    ap.add_argument("--gsm8k", default="data/gsm8k_train_sft.jsonl")
    ap.add_argument("--out", default="data/rft_correct.jsonl")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--max-keep", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    template = open(args.template).read()
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    eos = tok.eos_token_id
    stop_ids = sorted({eot, eos})
    print("stop_token_ids:", stop_ids, flush=True)

    rows = [json.loads(l) for l in open(args.gsm8k)]
    prompts_tok = []
    golds = []
    prompts_txt = []
    for r in rows:
        pstr = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True, chat_template=template,
        )
        ids = tok(pstr, add_special_tokens=False).input_ids  # template already has <bos>
        prompts_tok.append({"prompt_token_ids": ids})
        golds.append(norm(r["answer"]))
        prompts_txt.append(r["prompt"])

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=2048, enforce_eager=False, seed=args.seed)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=stop_ids, seed=args.seed)

    outs = llm.generate(prompts_tok, sp)

    n_q_solved = 0
    n_kept = 0
    with open(args.out, "w") as f:
        for i, o in enumerate(outs):
            gold = golds[i]
            seen = set()
            kept_here = 0
            for comp in o.outputs:
                text = comp.text.strip()
                m = ANS_RE.search(text)
                if not m:
                    continue
                if norm(m.group(1)) != gold:
                    continue
                # canonicalize: ensure single trailing marker + stop token
                body = text
                key = re.sub(r"\s+", " ", body)[:200]
                if key in seen:
                    continue
                seen.add(key)
                # strip any trailing <end_of_turn> the model emitted in text (usually stripped by stop)
                body = body.replace("<end_of_turn>", "").rstrip()
                if body.count("ANSWER:") != 1:
                    continue
                comp_out = body + "<end_of_turn>"
                f.write(json.dumps({
                    "prompt": MATH_PROMPT_TEMPLATE.format(prompt=prompts_txt[i]),
                    "completion": comp_out,
                    "answer": gold,
                    "src": "rft",
                }) + "\n")
                kept_here += 1
                n_kept += 1
                if kept_here >= args.max_keep:
                    break
            if kept_here > 0:
                n_q_solved += 1

    print(f"questions with >=1 correct sample: {n_q_solved}/{len(rows)}", flush=True)
    print(f"total kept correct solutions: {n_kept} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
