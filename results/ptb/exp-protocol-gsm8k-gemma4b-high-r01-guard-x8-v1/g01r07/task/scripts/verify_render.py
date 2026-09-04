"""Assert a training row's prompt is byte-identical to what the grader sends."""
import json, hashlib
from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot, MATH_PROMPT_TEMPLATE
from transformers import AutoTokenizer
SNAP="/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
tok = AutoTokenizer.from_pretrained(SNAP)
tok.chat_template = open("templates/gemma3.jinja").read()

rows=[json.loads(l) for l in open("data/sft_train.jsonl")]
z = next(r for r in rows if not r["kshot"]); k = next(r for r in rows if r["kshot"])

# what the grader would build for the same question
def grader_prompt(q, kshot):
    fs = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                    sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
    sysmsg = "\n\n".join([sample_to_fewshot(s) for s in fs])
    user = MATH_PROMPT_TEMPLATE.replace("{prompt}", q)
    msgs = ([{"role":"system","content":sysmsg}] if kshot else []) + [{"role":"user","content":user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

for r,name in ((z,"zero-shot"),(k,"10-shot")):
    g = grader_prompt(r["question"], bool(r["kshot"]))
    same = g == r["prompt"]
    print(f"{name}: byte-identical to grader render = {same}")
    assert same
    ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
    nbos = ids.count(tok.bos_token_id)
    print(f"  bos count in tokenized prompt = {nbos} (must be 1); first id {ids[0]}")
    assert nbos == 1
    cids = tok(r["completion"], add_special_tokens=False)["input_ids"]
    print(f"  target last id {cids[-1]} (106 = <end_of_turn>); 'ANSWER: ' count = {r['completion'].count('ANSWER: ')}")
    assert cids[-1] == 106 and r["completion"].count("ANSWER: ") == 1
# the 10-shot prompt the grader sends at eval time must equal the one we trained on
print("\nkshot prompt sha:", hashlib.sha256(k["prompt"].encode()).hexdigest()[:16])
print("ALL CHECKS PASS")
