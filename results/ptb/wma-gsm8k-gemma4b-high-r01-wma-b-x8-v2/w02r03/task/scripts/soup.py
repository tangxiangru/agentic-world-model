"""Weight-average two checkpoints of the same architecture (model soup).

exp-04 and exp-05 are consecutive points on one fine-tuning trajectory trained
on different mixtures; averaging them costs no GPU time and sometimes lands
above both. Config/tokenizer/generation_config come from --a.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on --a")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    idx = json.load(open(os.path.join(args.a, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    for sh in shards:
        ta = load_file(os.path.join(args.a, sh))
        tb = load_file(os.path.join(args.b, sh))
        assert set(ta) == set(tb), sh
        out = {}
        for k, v in ta.items():
            if v.is_floating_point():
                out[k] = (
                    args.alpha * v.float() + (1 - args.alpha) * tb[k].float()
                ).to(v.dtype)
            else:
                out[k] = v
        save_file(out, os.path.join(args.out, sh), metadata={"format": "pt"})
        print("wrote", sh, len(out), "tensors", flush=True)

    for f in os.listdir(args.a):
        s = os.path.join(args.a, f)
        d = os.path.join(args.out, f)
        if os.path.isdir(s) or f.endswith(".safetensors") or os.path.exists(d):
            continue
        if f in ("training_args.bin", "trainer_state.json"):
            continue
        shutil.copy(s, d)
    print("soup ->", args.out)


if __name__ == "__main__":
    main()
