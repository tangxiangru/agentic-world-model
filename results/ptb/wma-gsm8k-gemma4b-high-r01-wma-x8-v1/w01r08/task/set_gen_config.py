#!/usr/bin/env python3
"""Write the decoding defaults vLLM will pick up from a checkpoint's generation_config.json.

evaluate.py never sets a temperature, so vLLM falls back to the model's own
generation_config ("Default sampling parameters have been overridden by the
model's Hugging Face generation config"). For gemma-3-4b-pt that means
do_sample=true, top_k=64, top_p=0.95, i.e. the benchmark is scored on a SAMPLE.
"""
import json, sys, os

ckpt, mode = sys.argv[1], sys.argv[2]
p = os.path.join(ckpt, "generation_config.json")
cfg = json.load(open(p))
if mode == "greedy":
    cfg["do_sample"] = False
    cfg["temperature"] = 0.0
    cfg.pop("top_k", None)
    cfg.pop("top_p", None)
elif mode == "sample":
    cfg["do_sample"] = True
    cfg["temperature"] = 1.0
    cfg["top_k"] = 64
    cfg["top_p"] = 0.95
else:
    raise SystemExit("mode must be greedy|sample")
json.dump(cfg, open(p, "w"), indent=2)
print(p, json.dumps(cfg))
