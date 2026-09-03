"""Create a decode-config variant of a checkpoint without copying the weights.

evaluate.py -> vLLM reads generation_config.json from the model directory and
uses its temperature/top_p/top_k as the server's default sampling params
(vllm/config/model.py::get_diff_sampling_param). The base gemma-3-4b-pt file
ships top_k=64 / top_p=0.95 and no temperature, so the graded read is a
T=1.0 sample. This script hard-links the weights into a new directory and
rewrites only generation_config.json.
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
    for f in os.listdir(args.src):
        s, d = os.path.join(args.src, f), os.path.join(args.dst, f)
        if os.path.isdir(s) or os.path.exists(d):
            continue
        if f == "generation_config.json":
            continue
        try:
            os.link(s, d)
        except OSError:
            shutil.copy(s, d)

    gc = json.load(open(os.path.join(args.src, "generation_config.json")))
    gc["temperature"] = args.temperature
    for key, val in (("top_p", args.top_p), ("top_k", args.top_k)):
        if val is None:
            gc.pop(key, None)
        else:
            gc[key] = val
    gc["do_sample"] = args.temperature > 0
    with open(os.path.join(args.dst, "generation_config.json"), "w") as fh:
        json.dump(gc, fh, indent=2)
    print(json.dumps(gc, indent=2))
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
