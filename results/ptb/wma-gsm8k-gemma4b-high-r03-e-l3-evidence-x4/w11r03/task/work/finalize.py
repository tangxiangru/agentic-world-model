"""Assemble final_model/ from a chosen checkpoint and verify the grader can load it.

Guards the final_model_not_loadable pitfall: full weights (no adapter), tokenizer
alongside, greedy generation_config with eos_token_id [1, 106] intact, and a
from-scratch CPU load with transformers before anything is declared done.
"""
import argparse, json, os, shutil, sys

NEEDED = ["config.json", "model.safetensors.index.json", "tokenizer.json",
          "tokenizer_config.json", "special_tokens_map.json"]

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", default="/home/ben/task/final_model")
ap.add_argument("--tokenizer-from", default="/home/ben/task/ckpts/exp-02/final")
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--verify", action="store_true")
a = ap.parse_args()

os.makedirs(a.dst, exist_ok=True)
for f in os.listdir(a.dst):
    p = os.path.join(a.dst, f)
    os.remove(p) if os.path.isfile(p) else shutil.rmtree(p)

skip = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
        "training_args.bin", "generation_config.json"}
for f in os.listdir(a.src):
    if f in skip:
        continue
    s, d = os.path.join(a.src, f), os.path.join(a.dst, f)
    try:
        os.link(s, d)
    except OSError:
        shutil.copy(s, d)

for f in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
          "special_tokens_map.json", "added_tokens.json"]:
    s, d = os.path.join(a.tokenizer_from, f), os.path.join(a.dst, f)
    if os.path.exists(s) and not os.path.exists(d):
        os.link(s, d)

gc = json.load(open(os.path.join(a.tokenizer_from, "generation_config.json")))
gc.pop("top_p", None)
gc.pop("top_k", None)
gc["do_sample"] = a.temperature > 0
gc["temperature"] = a.temperature
json.dump(gc, open(os.path.join(a.dst, "generation_config.json"), "w"), indent=2)

assert gc["eos_token_id"] == [1, 106], gc["eos_token_id"]
missing = [f for f in NEEDED if not os.path.exists(os.path.join(a.dst, f))]
assert not missing, f"missing from final_model: {missing}"
print("files:", sorted(os.listdir(a.dst)))
print("generation_config:", json.dumps(gc))

if a.verify:
    import torch
    from transformers import AutoTokenizer, AutoConfig, Gemma3ForCausalLM
    cfg = AutoConfig.from_pretrained(a.dst)
    print("architectures:", cfg.architectures, "model_type:", cfg.model_type)
    tok = AutoTokenizer.from_pretrained(a.dst)
    m = Gemma3ForCausalLM.from_pretrained(a.dst, dtype=torch.bfloat16)
    n = sum(p.numel() for p in m.parameters())
    print(f"CPU load OK: {n/1e9:.3f}B params")
    tpl = open("/home/ben/task/templates/gemma3.jinja").read()
    r = tok.apply_chat_template([{"role": "user", "content": "2+2?"}],
                                chat_template=tpl, tokenize=False,
                                add_generation_prompt=True)
    print("template render OK:", repr(r))
    # evaluate.py's model_type() must resolve this directory to gemma3.jinja
    arch = cfg.architectures[0].lower()
    assert "gemma" in arch, arch
    print("evaluate.py will select templates/gemma3.jinja via architectures ->", arch)
