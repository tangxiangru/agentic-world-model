#!/usr/bin/env python3
"""Does the grader's 10-shot prefix help or hurt the SFT'd model?

Greedy-decodes the same held-out GSM8K *train* problems under 0-shot and 10-shot
prompts rendered by scripts/fmt.py. The eval always uses 10-shot, so a gap here
is a training-data design bug, not a benchmark property.
"""
import argparse, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt
from rft_sample import final_answer, norm_num

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", default=None)
a = ap.parse_args()

from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="train")
rows = []
for r in ds:
    g = norm_num(r["answer"].rsplit("####", 1)[-1])
    body = re.sub(r"<<[^>]*>>", "", r["answer"].rsplit("####", 1)[0]).strip()
    if g:
        rows.append({"q": r["question"].strip(), "gold": g, "reasoning": body})
rng = random.Random(a.seed)
rng.shuffle(rows)
probe, pool = rows[: a.n], rows[a.n:]

shots = rng.sample(pool, 10)
system = "\n\n".join(fmt.fewshot_block(s["q"], s["reasoning"], s["gold"]) for s in shots)

from vllm import LLM, SamplingParams
llm = LLM(model=a.model, gpu_memory_utilization=0.85, max_model_len=8192, dtype="bfloat16", seed=0)
sp = SamplingParams(temperature=0.0, max_tokens=768)

res = {}
for name, sysmsg in (("0shot", None), ("10shot", system)):
    outs = llm.generate([fmt.render(p["q"], system=sysmsg) for p in probe], sp)
    ok = sum(final_answer(o.outputs[0].text) == p["gold"] for o, p in zip(outs, probe))
    ntok = sum(len(o.outputs[0].token_ids) for o in outs) / len(probe)
    res[name] = {"acc": ok / len(probe), "n": len(probe), "mean_out_tokens": ntok}
    print(name, res[name], flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
