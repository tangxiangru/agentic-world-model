"""Copy a checkpoint into final_model/ and verify it the way the grader will load it."""
import argparse, json, os, shutil, sys, subprocess
ap=argparse.ArgumentParser(); ap.add_argument("--src",required=True); ap.add_argument("--dst",default="final_model")
a=ap.parse_args()
SNAP="/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
if os.path.exists(a.dst): shutil.rmtree(a.dst)
os.makedirs(a.dst)
for f in sorted(os.listdir(a.src)):
    if f in ("optimizer.pt","scheduler.pt","rng_state.pth","trainer_state.json","training_args.bin"): continue
    shutil.copy(os.path.join(a.src,f), os.path.join(a.dst,f))
# anything the trainer did not write, take from the base snapshot
for f in ("preprocessor_config.json","processor_config.json","tokenizer.json","tokenizer.model",
          "tokenizer_config.json","special_tokens_map.json","added_tokens.json"):
    if not os.path.exists(os.path.join(a.dst,f)) and os.path.exists(os.path.join(SNAP,f)):
        shutil.copy(os.path.join(SNAP,f), os.path.join(a.dst,f))
gc=os.path.join(a.dst,"generation_config.json")
cfg=json.load(open(gc)) if os.path.exists(gc) else {}
cfg.update({"bos_token_id":2,"eos_token_id":[1,106],"pad_token_id":0,"cache_implementation":"hybrid"})
json.dump(cfg,open(gc,"w"),indent=2)
print("files:", sorted(os.listdir(a.dst)))
cf=json.load(open(os.path.join(a.dst,"config.json")))
print("architectures:", cf["architectures"])
print("eos:", cfg["eos_token_id"])
# evaluate.py's model_type() must resolve; it lowercases the path first
print("model_type via path:", "gemma" if "gemma" in a.dst.lower() else "via config -> "+cf["architectures"][0].lower())
# load once on CPU from a fresh process, exactly the pitfall check asks for
r=subprocess.run([sys.executable,"-c",f"""
import torch, json
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
d="{a.dst}"
tok=AutoTokenizer.from_pretrained(d); print("tokenizer ok, vocab", len(tok))
cfg=AutoConfig.from_pretrained(d); print("config ok", cfg.model_type)
m=AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, device_map="cpu")
print("weights ok", type(m).__name__, sum(p.numel() for p in m.parameters())/1e9, "B params")
"""],capture_output=True,text=True)
print(r.stdout[-2000:]); print(r.stderr[-2000:] if r.returncode else "CPU LOAD OK")
sys.exit(r.returncode)
