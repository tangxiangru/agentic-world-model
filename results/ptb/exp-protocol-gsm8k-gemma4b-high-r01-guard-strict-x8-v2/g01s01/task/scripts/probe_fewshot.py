"""Diagnostic: does the harness's 10-shot system prefix help or hurt this model?

Scores the same GSM8K *train* questions (never test) greedily, twice: once with
the bare user turn the model was mostly trained on, and once with the exact
10-shot system prefix inspect_evals/gsm8k builds, rendered through the same
templates/gemma3.jinja path as the grader.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from datasets import load_dataset
from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402
from rft_sample import same  # noqa: E402

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def harness_system():
    fs = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                    sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
    return "\n\n".join(sample_to_fewshot(s) for s in fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--offset", type=int, default=6000)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split="train")
    qs = [{"q": ds[i]["question"].strip(),
           "a": ds[i]["answer"].rsplit("####", 1)[1].strip().replace(",", "")}
          for i in range(args.offset, min(args.offset + args.n, len(ds)))]
    sysmsg = harness_system()
    print(f"[probe] {len(qs)} train questions; 10-shot prefix is {len(sysmsg)} chars", flush=True)

    tok = AutoTokenizer.from_pretrained(SNAP)
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=4096)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=768, stop_token_ids=[1, 106])

    res = {}
    for name, sm in (("zero_shot", None), ("harness_10shot", sysmsg)):
        prompts = [TokensPrompt(prompt_token_ids=tok.encode(fmt.render_prompt(x["q"], sm),
                                                            add_special_tokens=False)) for x in qs]
        outs = llm.generate(prompts, sp)
        ok = sum(1 for x, o in zip(qs, outs) if same(o.outputs[0].text.strip(), x["a"]))
        trunc = sum(1 for o in outs if o.outputs[0].finish_reason != "stop")
        res[name] = {"accuracy": ok / len(qs), "n": len(qs), "truncated": trunc}
        print(f"[probe] {name}: {ok}/{len(qs)} = {ok/len(qs):.4f}  truncated={trunc}", flush=True)

    res["delta_10shot_minus_zeroshot"] = res["harness_10shot"]["accuracy"] - res["zero_shot"]["accuracy"]
    print(json.dumps(res, indent=2))
    with open("/home/ben/task/analysis/probe_fewshot.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
