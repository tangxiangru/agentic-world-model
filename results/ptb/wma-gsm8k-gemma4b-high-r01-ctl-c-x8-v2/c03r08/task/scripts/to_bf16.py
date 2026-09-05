#!/usr/bin/env python3
"""Cast a saved checkpoint to bfloat16 in place.

Trainer kept fp32 master weights, so final_model/ is 17.2 GB and vLLM's
dtype="auto" would load it in fp32 - which at evaluate.py's DEFAULT
--gpu-memory-utilization 0.3 (24 GB of an 80 GB card) leaves almost nothing for
the KV cache. The base checkpoint is bf16 and training ran under bf16 autocast,
so casting costs no meaningful precision and halves the footprint.
"""
import json, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

d = Path(sys.argv[1])
total = 0
for p in sorted(d.glob("*.safetensors")):
    sd = load_file(str(p))
    sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}
    save_file(sd, str(p), metadata={"format": "pt"})
    total += sum(v.numel() * v.element_size() for v in sd.values())
    print("cast", p.name)
    del sd

idx_path = d / "model.safetensors.index.json"
idx = json.loads(idx_path.read_text())
idx["metadata"]["total_size"] = total
idx_path.write_text(json.dumps(idx, indent=2))

cfg_path = d / "config.json"
cfg = json.loads(cfg_path.read_text())
cfg["torch_dtype"] = "bfloat16"
if "dtype" in cfg:
    cfg["dtype"] = "bfloat16"
for sub in ("text_config", "vision_config"):
    if isinstance(cfg.get(sub), dict) and "torch_dtype" in cfg[sub]:
        cfg[sub]["torch_dtype"] = "bfloat16"
cfg_path.write_text(json.dumps(cfg, indent=2))
print("total_size", total / 1e9, "GB; config dtype -> bfloat16")
