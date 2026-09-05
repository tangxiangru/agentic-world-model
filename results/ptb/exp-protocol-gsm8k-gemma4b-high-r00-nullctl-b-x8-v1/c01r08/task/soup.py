import os, json, shutil, torch
from safetensors.torch import load_file, save_file
from collections import defaultdict
import sys
srcs = sys.argv[1].split(",")
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
idx = json.load(open(os.path.join(srcs[0], "model.safetensors.index.json")))
files = sorted(set(idx["weight_map"].values()))
for fn in files:
    acc = None
    for s in srcs:
        sd = load_file(os.path.join(s, fn))
        if acc is None:
            acc = {k: v.to(torch.float32) for k, v in sd.items()}
        else:
            for k in acc: acc[k] += sd[k].to(torch.float32)
        del sd
    for k in acc: acc[k] = (acc[k] / len(srcs)).to(torch.bfloat16)
    save_file(acc, os.path.join(out, fn), metadata={"format": "pt"})
    print("wrote", fn)
    del acc
for f in ["model.safetensors.index.json","config.json","generation_config.json","tokenizer.json",
          "tokenizer.model","tokenizer_config.json","special_tokens_map.json","added_tokens.json",
          "preprocessor_config.json","processor_config.json","chat_template.jinja"]:
    p = os.path.join(srcs[0], f)
    if os.path.exists(p): shutil.copy(p, os.path.join(out, f))
print("done")
