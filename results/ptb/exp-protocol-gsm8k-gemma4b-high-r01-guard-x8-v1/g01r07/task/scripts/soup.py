"""Uniform weight average of two checkpoints from the same trajectory."""
import argparse, json, os, shutil, torch
from safetensors.torch import load_file, save_file
ap=argparse.ArgumentParser()
ap.add_argument("--a",required=True); ap.add_argument("--b",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--w",type=float,default=0.5,help="weight on --a")
x=ap.parse_args()
os.makedirs(x.out,exist_ok=True)
idx=json.load(open(os.path.join(x.a,"model.safetensors.index.json")))
shards=sorted(set(idx["weight_map"].values()))
n=0
for sh in shards:
    A=load_file(os.path.join(x.a,sh)); B=load_file(os.path.join(x.b,sh))
    assert set(A)==set(B), sh
    out={}
    for k in A:
        if A[k].is_floating_point():
            out[k]=(A[k].to(torch.float32)*x.w + B[k].to(torch.float32)*(1-x.w)).to(A[k].dtype)
            n+=1
        else:
            assert torch.equal(A[k],B[k]), k
            out[k]=A[k]
    save_file(out, os.path.join(x.out,sh), metadata={"format":"pt"})
    print("wrote",sh,flush=True)
for f in os.listdir(x.a):
    if f.endswith(".safetensors") or f in ("optimizer.pt","scheduler.pt","rng_state.pth","trainer_state.json","training_args.bin"): continue
    shutil.copy(os.path.join(x.a,f), os.path.join(x.out,f))
print(f"averaged {n} float tensors, w={x.w}; files:", sorted(os.listdir(x.out)))
