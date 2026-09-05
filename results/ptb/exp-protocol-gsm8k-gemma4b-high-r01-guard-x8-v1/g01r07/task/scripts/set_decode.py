"""Set the decode config of a saved checkpoint. Weights untouched."""
import argparse, json, os
ap=argparse.ArgumentParser(); ap.add_argument("--dir",required=True); ap.add_argument("--temperature",type=float,required=True)
a=ap.parse_args()
p=os.path.join(a.dir,"generation_config.json")
cfg=json.load(open(p)); before=dict(cfg)
cfg["temperature"]=a.temperature
if a.temperature==0.0:
    cfg["do_sample"]=False
    cfg.pop("top_k",None); cfg.pop("top_p",None)
cfg.update({"bos_token_id":2,"eos_token_id":[1,106],"pad_token_id":0,"cache_implementation":"hybrid"})
json.dump(cfg,open(p,"w"),indent=2)
print("before:",json.dumps(before)); print("after :",json.dumps(cfg))
