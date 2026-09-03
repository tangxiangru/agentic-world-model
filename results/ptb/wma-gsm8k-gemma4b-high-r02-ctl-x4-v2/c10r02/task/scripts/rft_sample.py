"""Rejection-sampling data generation: sample k solutions per question from a
fine-tuned checkpoint under the *grader's* prompt, keep the ones whose final
number matches gold.

Nothing here touches the GSM8K test split: questions come from the openai/gsm8k
TRAIN split and from OpenMathInstruct-2 rows that exp-02 did not train on.
"""
import argparse
import json
import os
import random

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

from inspect_ai.scorer._common import match_str

PROMPT_TEMPLATE = (
    'Solve the following math problem step by step. The last line of your response should be of the '
    'form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
    "{prompt}\n\n"
    'Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without '
    "quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.\n\n"
    "Reasoning:"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with {question, answer}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--fewshot", type=int, default=1, help="prepend the grader's 10-shot system message")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    template = open("/home/ben/task/templates/gemma3.jinja").read()
    fewshot = open("/home/ben/task/data/fewshot_system.txt").read()

    qs = [json.loads(l) for l in open(args.questions)]
    prompts = []
    for r in qs:
        msgs = []
        if args.fewshot:
            msgs.append({"role": "system", "content": fewshot})
        msgs.append({"role": "user", "content": PROMPT_TEMPLATE.format(prompt=r["question"])})
        prompts.append(tok.apply_chat_template(msgs, chat_template=template, tokenize=False,
                                               add_generation_prompt=True))
    print(f"{len(prompts)} prompts; example head:\n{prompts[0][:200]}", flush=True)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=4096,
              enable_prefix_caching=True, seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_tokens,
                        stop_token_ids=[tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id])
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    kept, n_corr, n_tot = [], 0, 0
    per_q_correct = []
    with open(args.out, "w") as fh:
        for r, o in zip(qs, outs):
            gold = r["answer"]
            cands = []
            for c in o.outputs:
                n_tot += 1
                text = c.text.strip()
                _, ok = match_str(value=text, target=gold, location="end", numeric=True)
                if not ok:
                    continue
                if text.count("ANSWER:") != 1 or not text.split("\n")[-1].startswith("ANSWER: "):
                    continue
                n_corr += 1
                cands.append(text)
            per_q_correct.append(len(cands))
            # prefer the shortest correct solutions (less rambling), dedup exact
            uniq = sorted(set(cands), key=len)
            for t in uniq[: args.max_per_question]:
                fh.write(json.dumps({"question": r["question"], "target": t,
                                     "answer": gold, "source": "rft"}) + "\n")
                kept.append(1)
    solved = sum(1 for c in per_q_correct if c > 0)
    print(json.dumps({
        "questions": len(qs), "samples": n_tot, "correct_samples": n_corr,
        "pass_rate": round(n_corr / max(1, n_tot), 4),
        "questions_with_at_least_one": solved,
        "solve_rate": round(solved / max(1, len(qs)), 4),
        "rows_written": len(kept), "out": args.out,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
