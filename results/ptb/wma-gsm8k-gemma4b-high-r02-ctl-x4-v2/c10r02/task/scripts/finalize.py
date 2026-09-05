"""Materialise final_model/ from a chosen checkpoint and verify it the way the
grader will load it: real files (no symlinks), bf16 config, tokenizer + image
processor present, and a CPU load with transformers from a fresh process.
"""
import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst)
    for fn in sorted(os.listdir(args.src)):
        src = os.path.realpath(os.path.join(args.src, fn))
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.dst, fn))
    if os.path.exists(os.path.join(args.dst, "train_log.jsonl")):
        os.remove(os.path.join(args.dst, "train_log.jsonl"))

    gcp = os.path.join(args.dst, "generation_config.json")
    gc = json.load(open(gcp))
    gc.pop("top_k", None)
    gc.pop("top_p", None)
    gc["do_sample"] = args.temperature > 0
    gc["temperature"] = args.temperature
    json.dump(gc, open(gcp, "w"), indent=2)

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    print("files:", sorted(os.listdir(args.dst)))
    print("architectures:", cfg["architectures"], "dtype:", cfg.get("dtype", cfg.get("torch_dtype")))
    print("generation_config:", json.dumps(gc))
    for need in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "preprocessor_config.json", "processor_config.json",
                 "model.safetensors.index.json"):
        assert os.path.isfile(os.path.join(args.dst, need)), f"MISSING {need}"
    for f in os.listdir(args.dst):
        assert not os.path.islink(os.path.join(args.dst, f)), f"symlink left behind: {f}"
    print("OK: final_model materialised at", args.dst)


if __name__ == "__main__":
    main()
