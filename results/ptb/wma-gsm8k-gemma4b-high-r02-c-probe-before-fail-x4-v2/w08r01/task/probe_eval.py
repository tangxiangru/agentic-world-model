"""Score a checkpoint on a held-out probe set built from the GSM8K TRAIN split.

Uses the same prompt, the same chat template and the same last-number grading
rule as evaluate.py, but on items the benchmark never sees, so it can be run at
any n without touching the test split. Also used to produce the watch-set result
that the experiment card asks for.
"""
from __future__ import annotations

import argparse
import json
import os
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from eval_format import fewshot_system_message, load_template, user_prompt

NUM = re.compile(r"-?\d+(?:[.,]\d+)*")


def last_number(text: str) -> str | None:
    words = re.split(r"\s+", text.strip())
    for w in reversed(words):
        m = NUM.findall(w.replace(",", ""))
        if m:
            try:
                v = float(m[-1])
            except ValueError:
                continue
            return str(int(v)) if v == int(v) else str(v)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/probe300.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=None,
                    help="omit to read temperature/top_p/top_k from the checkpoint's generation_config.json, "
                         "which is what the vLLM server does for evaluate.py")
    ap.add_argument("--n", type=int, default=1, help="samples per item (>1 = pass@k / maj@k data)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default=None, help="write per-item generations here")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    tpl = load_template()
    sysmsg = fewshot_system_message()
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": sysmsg},
             {"role": "user", "content": user_prompt(r["question"])}],
            chat_template=tpl, tokenize=False, add_generation_prompt=True,
        )
        for r in rows
    ]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=4096,
              enforce_eager=False, disable_log_stats=True)
    # vLLM's offline LLM.generate uses the SamplingParams it is handed verbatim; it does NOT
    # fall back to generation_config. Replicate the server's behaviour explicitly.
    sp_kwargs = {"max_tokens": args.max_tokens, "n": args.n}
    if args.temperature is None:
        gc = json.load(open(os.path.join(args.model, "generation_config.json")))
        sp_kwargs["temperature"] = gc.get("temperature", 1.0)
        if "top_p" in gc:
            sp_kwargs["top_p"] = gc["top_p"]
        if "top_k" in gc:
            sp_kwargs["top_k"] = gc["top_k"]
    else:
        sp_kwargs["temperature"] = args.temperature
    if sp_kwargs["temperature"] == 0:
        sp_kwargs["top_p"] = 1.0
        sp_kwargs["top_k"] = -1
    print("sampling params:", sp_kwargs)
    sp = SamplingParams(**sp_kwargs)
    outs = llm.generate(prompts, sp)

    n_ok = 0
    recs = []
    for r, o in zip(rows, outs):
        texts = [c.text for c in o.outputs]
        preds = [last_number(t) for t in texts]
        gold = r["gold"]
        ok = preds[0] == gold
        n_ok += ok
        recs.append({"id": r["id"], "gold": gold, "pred": preds[0], "correct": bool(ok),
                     "preds": preds, "n_chars": len(texts[0]),
                     "last_line": texts[0].strip().splitlines()[-1].strip() if texts[0].strip() else ""})
    acc = n_ok / len(rows)
    summary = {"model": args.model, "data": args.data, "n": len(rows), "accuracy": acc,
               "temperature": args.temperature, "samples_per_item": args.n}
    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "items": recs}, f, indent=2)
    if args.dump:
        with open(args.dump, "w") as f:
            for r, o in zip(rows, outs):
                f.write(json.dumps({"id": r["id"], "gold": r["gold"],
                                    "generations": [c.text for c in o.outputs]}) + "\n")


if __name__ == "__main__":
    main()
