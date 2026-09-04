"""Write the decode defaults the grader will use.

evaluate.py sends no temperature, so vLLM falls back to the model directory's
generation_config.json ("Default sampling parameters have been overridden by the
model's Hugging Face generation config", logs/exp-01.log). That file is part of
the checkpoint we ship, so it is where the decode policy is set.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--mode", choices=["greedy", "base"], default="greedy")
args = ap.parse_args()

p = os.path.join(args.model, "generation_config.json")
cfg = json.load(open(p))
if args.mode == "greedy":
    cfg.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0})
else:
    cfg.update({"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64})
json.dump(cfg, open(p, "w"), indent=2)
print(json.dumps(cfg, indent=2))
