"""Assemble final_model/ from a chosen checkpoint and verify it loads.

The grader loads final_model/ with vLLM from a fresh process and picks the
chat template from config.json['architectures'] (evaluate.py::model_type), so
this copies real files (no symlinks), keeps the tokenizer/processor next to
the weights, and writes the greedy generation_config adopted in exp-03.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, "/home/ben/task/scripts")
import render  # noqa: E402

NEEDED = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst)
    for f in sorted(os.listdir(args.src)):
        s = os.path.join(args.src, f)
        if os.path.isdir(s) or f in ("training_args.bin", "trainer_state.json"):
            continue
        shutil.copy(s, os.path.join(args.dst, f))
        print("copied", f, flush=True)

    # tokenizer files can be missing from a mid-training checkpoint dir
    for f in NEEDED:
        d = os.path.join(args.dst, f)
        if not os.path.exists(d):
            src = os.path.join(render.MODEL_PATH, f)
            shutil.copy(src, d)
            print("filled from base:", f, flush=True)

    gc = json.load(open(os.path.join(args.dst, "generation_config.json")))
    gc["temperature"] = args.temperature
    gc["do_sample"] = args.temperature > 0
    gc.pop("top_p", None)
    gc.pop("top_k", None)
    eos = gc.get("eos_token_id")
    assert eos == [1, 106] or (isinstance(eos, list) and 106 in eos), f"eos {eos}"
    json.dump(gc, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)
    print("generation_config:", json.dumps(gc))

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    arch = cfg["architectures"][0]
    assert "gemma" in arch.lower(), arch
    print("architectures:", cfg["architectures"], "-> evaluate.py picks gemma3.jinja")

    # load once on CPU exactly as a fresh process would
    from transformers import AutoConfig, AutoTokenizer

    AutoConfig.from_pretrained(args.dst)
    tok = AutoTokenizer.from_pretrained(args.dst)
    ids = tok(render.render_target("2 + 2 = 4.", "4"), add_special_tokens=False).input_ids
    assert ids[-1] == 106, ids[-1]
    print("tokenizer ok; target ends on token 106 <end_of_turn>")

    import safetensors.torch as st

    idx = json.load(open(os.path.join(args.dst, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    n = 0
    for sh in shards:
        with st.safe_open(os.path.join(args.dst, sh), framework="pt") as fh:
            n += len(fh.keys())
    print(f"{len(shards)} shards, {n} tensors, index lists {len(idx['weight_map'])}")
    assert n == len(idx["weight_map"]), "weight map / shard mismatch"
    print("OK ->", args.dst)


if __name__ == "__main__":
    main()
