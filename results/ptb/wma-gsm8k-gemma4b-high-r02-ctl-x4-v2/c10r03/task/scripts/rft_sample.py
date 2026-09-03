"""Rejection-sampling data generation with vLLM, offline.

Samples k solutions per GSM8K *train* question from a checkpoint, keeps the ones
whose final ANSWER matches the reference answer, dedups, and writes rows in the
same schema as data/sft_v1.jsonl (question / target / src) so the SFT trainer
reads them unchanged. Also usable as a fast local scorer (--score-only) against
the 500 held-out train items.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from promptlib import (  # noqa: E402
    STOP_TOKEN,
    fewshot_system_message,
    load_chat_template,
    user_prompt,
)

NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def norm(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    if not math.isfinite(f):
        return None
    return str(int(f)) if f == int(f) and abs(f) < 1e15 else "%g" % f


def graded_answer(completion: str) -> str | None:
    """Mimic inspect_ai match(location='end', numeric=True): the LAST number."""
    nums = NUMBER.findall(completion)
    return norm(nums[-1]) if nums else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", default=None,
                    help="jsonl of {question, gold}; default = GSM8K train minus holdout")
    ap.add_argument("--out", default="/home/ben/task/data/rft.jsonl")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fewshot", action="store_true",
                    help="put the grader's 10-shot system prefix in the prompt")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    items = []
    if a.questions:
        with open(a.questions) as f:
            for line in f:
                r = json.loads(line)
                items.append({"question": r["question"], "gold": norm(str(r["gold"]))})
    else:
        from datasets import load_dataset

        gsm = load_dataset("openai/gsm8k", "main")["train"]
        n = len(gsm)
        for i in range(n - 500):  # last 500 are the held-out probe set
            r = gsm[i]
            items.append({"question": r["question"].strip(),
                          "gold": norm(r["answer"].split("####")[-1])})
    if a.limit:
        items = items[: a.limit]
    print(f"{len(items)} questions", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.model)
    tpl = load_chat_template()
    sys_msg = fewshot_system_message() if a.fewshot else None

    prompts = []
    for it in items:
        msgs = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + [
            {"role": "user", "content": user_prompt(it["question"])}
        ]
        prompts.append(
            tok.apply_chat_template(msgs, chat_template=tpl, tokenize=False,
                                    add_generation_prompt=True)
        )

    k = 1 if a.score_only else a.k
    sp = SamplingParams(
        n=k,
        temperature=0.0 if a.score_only else a.temperature,
        top_p=1.0 if a.score_only else a.top_p,
        max_tokens=a.max_tokens,
        stop_token_ids=[1, 106],
        seed=a.seed,
    )
    llm = LLM(
        model=a.model,
        gpu_memory_utilization=a.gpu_mem,
        max_model_len=4096,
        dtype="bfloat16",
        enforce_eager=False,
        disable_log_stats=True,
    )
    outs = llm.generate(prompts, sp)

    if a.score_only:
        ok = 0
        recs = []
        for it, o in zip(items, outs):
            c = o.outputs[0].text
            got = graded_answer(c)
            hit = got is not None and got == it["gold"]
            ok += hit
            recs.append({"question": it["question"], "gold": it["gold"],
                         "pred": got, "correct": bool(hit), "completion": c})
        print(f"accuracy {ok}/{len(items)} = {ok/len(items):.4f}", flush=True)
        with open(a.out, "w") as f:
            json.dump({"accuracy": ok / len(items), "n": len(items),
                       "samples": recs}, f)
        return

    kept, seen_pass, n_any = 0, 0, 0
    with open(a.out, "w") as f:
        for it, o in zip(items, outs):
            good, texts = [], set()
            for cand in o.outputs:
                c = cand.text.strip()
                if not c or graded_answer(c) != it["gold"]:
                    continue
                if "ANSWER:" not in c or c.count("ANSWER:") != 1:
                    continue
                if not c.rstrip().endswith(it["gold"]) and \
                        not re.search(r"ANSWER:\s*-?[\d,.]+\s*$", c):
                    continue
                if c in texts:
                    continue
                texts.add(c)
                good.append(c)
            if good:
                seen_pass += 1
            good.sort(key=len)  # prefer the shorter correct chain
            for c in good[: a.max_per_question]:
                f.write(json.dumps({"question": it["question"],
                                    "target": c + STOP_TOKEN,
                                    "src": "rft"}) + "\n")
                kept += 1
            n_any += 1
    print(f"questions with >=1 correct sample: {seen_pass}/{n_any} "
          f"({seen_pass/n_any:.3f}); rows written: {kept}", flush=True)


if __name__ == "__main__":
    main()
