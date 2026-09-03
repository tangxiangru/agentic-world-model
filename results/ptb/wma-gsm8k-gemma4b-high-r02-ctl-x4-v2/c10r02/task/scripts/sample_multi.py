"""Sample k candidate solutions per question and keep ALL of them, with a
correctness flag, so that a majority-vote target can be built from real
disagreement rather than from k copies of the right answer.

Questions come from the openai/gsm8k TRAIN split and from OpenMathInstruct-2;
never from the gsm8k test split.
"""
import argparse
import json
import os
import re

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
ANS_RE = re.compile(r"^ANSWER:\s*(-?[\d,]*\.?\d+)\s*$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    template = open("/home/ben/task/templates/gemma3.jinja").read()
    qs = [json.loads(l) for l in open(args.questions)]
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(prompt=r["question"])}],
            chat_template=template, tokenize=False, add_generation_prompt=True)
        for r in qs
    ]
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=2048,
              enable_prefix_caching=True, seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_tokens,
                        stop_token_ids=[tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id])
    outs = llm.generate(prompts, sp)

    n_ok = n_tot = 0
    with open(args.out, "w") as fh:
        for r, o in zip(qs, outs):
            cands = []
            for c in o.outputs:
                n_tot += 1
                text = c.text.strip()
                last = text.split("\n")[-1].strip()
                m = ANS_RE.match(last)
                if m is None or text.count("ANSWER:") != 1:
                    continue
                pred = m.group(1).replace(",", "")
                _, ok = match_str(value=text, target=r["answer"], location="end", numeric=True)
                n_ok += int(ok)
                cands.append({"body": text[: text.rfind("\nANSWER:")].strip(), "pred": pred,
                              "correct": bool(ok)})
            if cands:
                fh.write(json.dumps({"question": r["question"], "answer": r["answer"],
                                     "cands": cands}) + "\n")
    print(json.dumps({"questions": len(qs), "samples": n_tot, "correct": n_ok,
                      "pass_rate": round(n_ok / max(1, n_tot), 4), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
