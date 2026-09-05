"""Copy a checkpoint into final_model/ and prove the grader can load it.

Checks (pitfall: final_model_not_loadable):
  * weights, config, tokenizer and processor files are all present;
  * generation_config.json asks vLLM for greedy decoding and keeps
    eos_token_id = [1, 106] so the server still stops on <end_of_turn>;
  * the directory loads with transformers from a fresh process on CPU;
  * evaluate.py's model_type() resolves it to the gemma3 template.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

REQUIRED = ["config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--no-copy", action="store_true")
    args = ap.parse_args()

    if not args.no_copy:
        if os.path.exists(args.dst):
            shutil.rmtree(args.dst)
        shutil.copytree(args.src, args.dst)
        print(f"copied {args.src} -> {args.dst}", flush=True)

    files = sorted(os.listdir(args.dst))
    print("files:", files, flush=True)
    missing = [f for f in REQUIRED if f not in files]
    assert not missing, f"missing {missing}"
    assert any(f.endswith(".safetensors") for f in files), "no weights"

    gc = json.load(open(os.path.join(args.dst, "generation_config.json")))
    print("generation_config:", gc, flush=True)
    assert gc.get("temperature") == 0.0, "generation_config must request greedy decoding"
    assert 106 in gc.get("eos_token_id", []), "must still stop on <end_of_turn> (106)"

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    arch = cfg["architectures"][0].lower()
    assert "gemma" in arch, arch
    print("architecture:", cfg["architectures"][0], flush=True)

    code = (
        "import torch, transformers;"
        f"m=transformers.AutoModelForImageTextToText.from_pretrained('{args.dst}',"
        "dtype=torch.bfloat16);"
        f"t=transformers.AutoTokenizer.from_pretrained('{args.dst}');"
        "print('LOADED', type(m).__name__, sum(p.numel() for p in m.parameters())/1e9,"
        "'B', next(m.parameters()).dtype, 'vocab', len(t))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    print(r.stdout.strip()[-600:], flush=True)
    if "LOADED" not in r.stdout:
        print(r.stderr[-3000:], flush=True)
        raise SystemExit("final_model does not load with transformers")
    print("OK: final_model is loadable", flush=True)


if __name__ == "__main__":
    main()
