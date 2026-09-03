import json, random, re, sys
from build_data import MATH_PROMPT_TEMPLATE, CALC_RE, normalize_answer, fewshot_block
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams

tmpl = open("templates/gemma3.jinja").read()
M = "ckpts/exp-02/final"
tok = AutoTokenizer.from_pretrained(M)
gsm = load_dataset("openai/gsm8k", "main", split="train")
rows = list(gsm)[:200]
shot_pool = []
for r in list(gsm)[200:400]:
    q = r["question"].strip(); b, _, t = r["answer"].rpartition("####")
    shot_pool.append((q, CALC_RE.sub("", b).strip(), t.strip()))

probs = [(r["question"].strip(), normalize_answer(r["answer"].rpartition("####")[2])) for r in rows]
rng = random.Random(0)


def mk(q, nshot):
    msgs = []
    if nshot:
        msgs.append({"role": "system", "content": fewshot_block(rng.sample(shot_pool, nshot))})
    msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)})
    return tok.apply_chat_template(msgs, chat_template=tmpl, tokenize=False, add_generation_prompt=True)


llm = LLM(model=M, gpu_memory_utilization=0.85, max_model_len=4096, dtype="bfloat16")
ANS = re.compile(r"ANSWER:\s*(\$?-?[\d,]*\.?\d+)\s*$")


def run(label, nshot, sp):
    prompts = [mk(q, nshot) for q, a in probs]
    outs = llm.generate(prompts, sp)
    ok = tot = 0
    nolast = 0
    lens = []
    for o, (q, a) in zip(outs, probs):
        for c in o.outputs:
            tot += 1
            lens.append(len(c.token_ids))
            m = ANS.search(c.text.strip())
            if not m:
                nolast += 1
            elif normalize_answer(m.group(1)) == a:
                ok += 1
    lens.sort()
    print(f"{label}: {ok}/{tot} = {ok/tot:.3f}  no-ANSWER-at-end={nolast}  "
          f"len p50={lens[len(lens)//2]} p95={lens[int(len(lens)*0.95)]} max={lens[-1]}", flush=True)
    return outs


o3 = run("t1.0 seeded n4", 0, SamplingParams(n=4, temperature=1.0, top_p=0.95, max_tokens=1024, seed=0))
o4 = run("t1.0 unseeded n4", 0, SamplingParams(n=4, temperature=1.0, top_p=0.95, max_tokens=1024))
o5 = run("t0.8 unseeded n4", 0, SamplingParams(n=4, temperature=0.8, top_p=0.95, max_tokens=1024))
print("uniq of seeded n4 first 5:", [len(set(c.text for c in o.outputs)) for o in o3[:5]], flush=True)
print("uniq of unseeded n4 first 5:", [len(set(c.text for c in o.outputs)) for o in o4[:5]], flush=True)
