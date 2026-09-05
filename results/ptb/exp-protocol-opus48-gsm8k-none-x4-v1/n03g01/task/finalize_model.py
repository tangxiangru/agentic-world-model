#!/usr/bin/env python3
"""Copy processor configs from base snapshot and set greedy generation config."""
import os, json, shutil, sys

BASE = os.environ["PTB_BASE_MODEL_SNAPSHOT"]

def finalize(d):
    for f in ["preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(BASE, f)
        dst = os.path.join(d, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print("copied", f)
    gc_path = os.path.join(d, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
    gc["do_sample"] = False
    gc["temperature"] = 0.0
    gc.pop("top_p", None); gc.pop("top_k", None)
    gc.setdefault("bos_token_id", 2)
    gc.setdefault("eos_token_id", [1, 106])
    gc.setdefault("pad_token_id", 0)
    json.dump(gc, open(gc_path, "w"), indent=2)
    print("greedy gen config set for", d, gc)

if __name__ == "__main__":
    finalize(sys.argv[1])
