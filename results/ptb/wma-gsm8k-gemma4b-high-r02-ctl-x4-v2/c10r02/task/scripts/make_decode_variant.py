"""Create a checkpoint directory that differs from its parent only in
generation_config.json.

vLLM's ModelConfig.get_diff_sampling_param() reads generation_config.json
(generation_config="auto" is the default) and uses temperature / top_p / top_k
as the server's default sampling params for any request that does not set them.
evaluate.py sets only max_tokens, so this file is what decides how the graded
completion is decoded.

Weight files are symlinked, so a variant costs no disk.
"""
import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for fn in sorted(os.listdir(args.src)):
        src = os.path.join(args.src, fn)
        dst = os.path.join(args.dst, fn)
        if os.path.lexists(dst):
            os.remove(dst)
        if fn.endswith(".safetensors"):
            os.symlink(os.path.abspath(src), dst)
        elif os.path.isfile(src):
            shutil.copy(src, dst)

    gc_path = os.path.join(args.dst, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
    gc.pop("top_k", None)
    gc.pop("top_p", None)
    gc["do_sample"] = args.temperature > 0
    gc["temperature"] = args.temperature
    if args.top_p is not None:
        gc["top_p"] = args.top_p
    if args.top_k is not None:
        gc["top_k"] = args.top_k
    json.dump(gc, open(gc_path, "w"), indent=2)
    print(json.dumps(gc, indent=2))
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
