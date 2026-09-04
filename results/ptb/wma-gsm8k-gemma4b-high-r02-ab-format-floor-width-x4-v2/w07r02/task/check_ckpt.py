#!/usr/bin/env python3
"""WMA preconditions 1-2: the moment a checkpoint lands, prove it is servable."""
import json, os, sys, time
SNAP="/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
ck=sys.argv[1]
while not os.path.exists(os.path.join(ck,"config.json")):
    time.sleep(20)
time.sleep(60)  # let the save finish
out={"ckpt":ck}
gc=json.load(open(os.path.join(ck,"generation_config.json")))
base=json.load(open(os.path.join(SNAP,"generation_config.json")))
out["generation_config"]=gc
out["base_generation_config"]=base
out["eos_ok"]= gc.get("eos_token_id")==[1,106]
out["topk_ok"]= gc.get("top_k")==64 and gc.get("top_p")==0.95
cfg=json.load(open(os.path.join(ck,"config.json")))
out["architectures"]=cfg.get("architectures")
out["files"]=sorted(os.listdir(ck))
try:
    from transformers import AutoTokenizer, AutoConfig
    AutoTokenizer.from_pretrained(ck); AutoConfig.from_pretrained(ck)
    out["tokenizer_config_load"]="ok"
except Exception as e:
    out["tokenizer_config_load"]=f"FAIL {e}"
print(json.dumps(out, indent=2))
