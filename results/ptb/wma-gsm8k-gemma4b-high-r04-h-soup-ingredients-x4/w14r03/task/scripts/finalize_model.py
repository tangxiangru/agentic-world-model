#!/usr/bin/env python3
"""Copy a checkpoint into final_model/ as real files and verify it loads.

Guards the `final_model_not_loadable` pitfall: real shards (never symlinks),
tokenizer + multimodal side-cars present, greedy generation_config with
<end_of_turn> still in eos_token_id, bfloat16 dtype, and a CPU load with
transformers from a fresh process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GREEDY = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "transformers_version": "4.50.0.dev0",
}
NEEDED = [
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "preprocessor_config.json", "processor_config.json",
]


def sha256(path: str, limit: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--skip-copy", action="store_true")
    ap.add_argument("--score", type=float, default=None,
                    help="dev-150 accuracy of --src; refuses to overwrite a better final_model")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    prev = "/home/ben/task/analysis/final_model_check.json"
    if not a.skip_copy and not a.force and a.score is not None and os.path.exists(prev):
        old = json.load(open(prev)).get("score")
        if old is not None and a.score < old:
            print(f"refusing to overwrite: final_model holds {old}, --src scores {a.score}")
            sys.exit(2)
    if not a.skip_copy:
        if os.path.exists(a.dst):
            shutil.rmtree(a.dst)
        os.makedirs(a.dst)
        for fn in sorted(os.listdir(a.src)):
            if fn.startswith("checkpoint-") or fn == "training_args.bin":
                continue
            src = os.path.realpath(os.path.join(a.src, fn))
            if os.path.isdir(src):
                continue
            shutil.copyfile(src, os.path.join(a.dst, fn))  # real bytes, not a link
        for fn in ("preprocessor_config.json", "processor_config.json",
                   "tokenizer.model", "added_tokens.json"):
            d = os.path.join(a.dst, fn)
            if not os.path.exists(d):
                s = os.path.join(common.BASE_MODEL, fn)
                if os.path.exists(s):
                    shutil.copyfile(s, d)
        json.dump(GREEDY, open(os.path.join(a.dst, "generation_config.json"), "w"), indent=2)

    report = {"dst": a.dst, "src": a.src, "score": a.score, "files": {}, "checks": {}}
    for fn in sorted(os.listdir(a.dst)):
        p = os.path.join(a.dst, fn)
        report["files"][fn] = {
            "bytes": os.path.getsize(p),
            "is_symlink": os.path.islink(p),
            "sha256_16": sha256(p),
        }
    report["checks"]["no_symlinks"] = not any(v["is_symlink"] for v in report["files"].values())
    report["checks"]["required_files"] = [f for f in NEEDED if f not in report["files"]] or "all present"
    report["checks"]["has_weights"] = any(f.endswith(".safetensors") for f in report["files"])

    cfg = json.load(open(os.path.join(a.dst, "config.json")))
    gc = json.load(open(os.path.join(a.dst, "generation_config.json")))
    report["checks"]["dtype"] = cfg.get("dtype") or cfg.get("torch_dtype")
    report["checks"]["eos_has_end_of_turn"] = 106 in gc["eos_token_id"]
    report["checks"]["greedy"] = gc.get("temperature") == 0.0

    # a fresh-process CPU load, the way the grader's vLLM will read it
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.dst)
    m = AutoModelForCausalLM.from_pretrained(a.dst, dtype=torch.bfloat16)
    report["checks"]["params_M"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 1)
    report["checks"]["model_dtype"] = str(next(m.parameters()).dtype)
    report["checks"]["tokenizer_len"] = len(tok)
    report["checks"]["decodes_end_of_turn"] = tok.decode([106])

    json.dump(report, open("/home/ben/task/analysis/final_model_check.json", "w"), indent=2)
    print(json.dumps(report["checks"], indent=2))
    bad = [k for k, v in report["checks"].items() if v is False]
    if bad or report["checks"]["required_files"] != "all present":
        print("FAILED CHECKS:", bad, report["checks"]["required_files"])
        sys.exit(1)
    print("final_model OK")


if __name__ == "__main__":
    main()
