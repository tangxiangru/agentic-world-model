import torch, os, json, shutil, glob
from safetensors.torch import load_file, save_file
srcs = ["runs/sft_v2", "runs/sft_v3"]
dst = "runs/soup"
os.makedirs(dst, exist_ok=True)
idx = json.load(open(os.path.join(srcs[0], "model.safetensors.index.json")))
files = sorted(set(idx["weight_map"].values()))
for fn in files:
    acc = None
    for s in srcs:
        sd = load_file(os.path.join(s, fn))
        if acc is None:
            acc = {k: v.to(torch.float32) for k, v in sd.items()}
        else:
            for k in acc:
                acc[k] += sd[k].to(torch.float32)
    out = {k: (v / len(srcs)).to(torch.bfloat16) for k, v in acc.items()}
    save_file(out, os.path.join(dst, fn), metadata={"format": "pt"})
    print("wrote", fn)
for b in ["model.safetensors.index.json","config.json","added_tokens.json","tokenizer.json","tokenizer.model",
          "tokenizer_config.json","special_tokens_map.json","preprocessor_config.json","processor_config.json"]:
    p = os.path.join(srcs[0], b)
    if os.path.exists(p):
        shutil.copy(os.path.realpath(p), os.path.join(dst, b))
shutil.copy("final_model/generation_config.json", os.path.join(dst, "generation_config.json"))
print("soup done")
