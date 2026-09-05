import json, re, random, os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

random.seed(0)
SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
tok = AutoTokenizer.from_pretrained(SNAP)
tok.chat_template = open("templates/gemma3.jinja").read()

rows = [json.loads(l) for l in open("train_data.jsonl")]
# recover question/target from stored fields
prompts, meta = [], []
for r in rows:
    tgt = r["assistant"].split("ANSWER:")[-1].strip()
    p = tok.apply_chat_template([{"role": "user", "content": r["user"]}],
                                tokenize=False, add_generation_prompt=True)
    prompts.append(p)
    meta.append({"user": r["user"], "target": tgt})

N = int(os.environ.get("NQ", 7000))
prompts, meta = prompts[:N], meta[:N]

llm = LLM(model="model_v1", dtype="bfloat16", gpu_memory_utilization=0.9,
          max_model_len=1024, enable_prefix_caching=True)
sp = SamplingParams(n=4, temperature=1.0, top_p=0.95, max_tokens=380,
                    stop=["<end_of_turn>"])
outs = llm.generate(prompts, sp)

def norm(x):
    x = x.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return None

kept = []
nq_solved = 0
for o, m in zip(outs, meta):
    good, seen = [], set()
    for c in o.outputs:
        t = c.text.strip()
        if "ANSWER:" not in t:
            continue
        pred = norm(t.split("ANSWER:")[-1].split("\n")[0])
        if pred is None or pred != norm(m["target"]):
            continue
        body = t.split("ANSWER:")[0].strip()
        if len(body) < 15 or body in seen:
            continue
        # reject degenerate/rambling solutions
        if len(body) > 1200:
            continue
        seen.add(body)
        good.append(t)
    if good:
        nq_solved += 1
    for t in good[:2]:
        kept.append({"system": None, "user": m["user"], "assistant": t})

print("questions with >=1 correct:", nq_solved, "/", len(meta), "kept samples:", len(kept))
with open("rft_data.jsonl", "w") as f:
    for k in kept:
        f.write(json.dumps(k) + "\n")
with open("rft_decon.jsonl", "w") as f:
    for k in kept:
        f.write(json.dumps({"text": k["assistant"]}) + "\n")
