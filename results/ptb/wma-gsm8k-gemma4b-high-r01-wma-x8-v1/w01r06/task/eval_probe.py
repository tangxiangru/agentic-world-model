#!/usr/bin/env python3
"""Cheap private probe: score a checkpoint on data/dev250.jsonl.

dev250 is 250 items held out of the GSM8K *train* split and excluded from
every training file, so this can be run as often as needed without touching
the benchmark test set. It reproduces the harness end to end:
  * the same 10-shot system block (data/fewshot_system.txt, lifted verbatim
    from the inspect log of exp-01),
  * the same MATH_PROMPT_TEMPLATE user message,
  * templates/gemma3.jinja as the chat template,
  * inspect's match(numeric=True, location="end") rule: the LAST number in
    the completion, normalised, must equal the gold answer.

--decode default reproduces what evaluate.py actually gets: vLLM reads the
model's generation_config.json, so unless it says otherwise the harness
samples at temperature 1.0 / top_k 64 / top_p 0.95.
"""
from __future__ import annotations

import argparse
import json
import re

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def strip_numeric_punctuation(s: str) -> str:
    """Mirror inspect_ai._util.text.strip_numeric_punctuation closely enough."""
    s = s.strip()
    s = re.sub(r"^[$£€¥]", "", s)
    s = s.rstrip(".,;:!?)")
    s = s.lstrip("(")
    return s.replace(",", "")


def last_number(text: str) -> str | None:
    words = re.split(r"\s+", strip_numeric_punctuation(text.strip().casefold()))
    for w in reversed(words):
        w = strip_numeric_punctuation(w)
        if w.replace(".", "").replace("-", "").isnumeric() and any(c.isdigit() for c in w):
            return w
    return None


def norm_num(s: str) -> str | None:
    try:
        return format(float(s), ".5g")
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/dev250.jsonl")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--decode", choices=["greedy", "harness"], default="harness")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    ap.add_argument("--shots", type=int, default=10,
                    help="10 = the harness's own system block; 0 = no system "
                         "message (the format training used); 1-9 = first k of "
                         "the harness's blocks")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tpl = open("templates/gemma3.jinja").read()
    system = open("data/fewshot_system.txt").read()
    rows = [json.loads(l) for l in open(args.data)][: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)

    if args.shots < 10:
        blocks = re.split(r"(?<=\n)\n(?=[^\n])", system)
        blocks = [b for b in system.split("\n\n") if b.strip()]
        # rebuild by shot: each shot is "question\n\nReasoning:\n...\n\nANSWER: n"
        shots = re.split(r"(?<=\n)(?=[^\n])", system)
        pieces, cur = [], []
        for line in system.split("\n"):
            cur.append(line)
            if line.startswith("ANSWER: "):
                pieces.append("\n".join(cur).strip())
                cur = []
        system = "\n\n".join(pieces[: args.shots])

    def render(question: str) -> str:
        msgs = []
        if args.shots > 0:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user",
                     "content": PROMPT_TEMPLATE.format(prompt=question)})
        return tok.apply_chat_template(msgs, chat_template=tpl, tokenize=False,
                                       add_generation_prompt=True)

    prompts = [render(r["question"]) for r in rows]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util,
              max_model_len=8192, enforce_eager=False)
    if args.decode == "greedy":
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    else:
        sp = SamplingParams(temperature=1.0, top_k=64, top_p=0.95,
                            max_tokens=args.max_tokens)
    outs = llm.generate(prompts, sp)

    n_ok = n_stop = 0
    recs = []
    for r, o in zip(rows, outs):
        text = o.outputs[0].text
        pred = last_number(text)
        gold = norm_num(strip_numeric_punctuation(r["gold"].casefold()))
        ok = pred is not None and norm_num(pred) == gold
        stopped = o.outputs[0].finish_reason == "stop"
        n_ok += ok
        n_stop += stopped
        recs.append({"id": r["id"], "gold": r["gold"], "pred": pred,
                     "correct": bool(ok), "stopped": stopped, "text": text})

    n = len(rows)
    summary = {"model": args.model, "decode": args.decode, "shots": args.shots,
               "n": n, "accuracy": n_ok / n, "stop_rate": n_stop / n}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({**summary, "samples": recs}, f)


if __name__ == "__main__":
    main()
