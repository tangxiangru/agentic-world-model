#!/usr/bin/env python3
"""Make a Trainer checkpoint dir servable by the grader's evaluate.py:
tokenizer + processor files alongside the weights, and a greedy
generation_config vLLM will actually honour.

vLLM's ModelConfig.get_diff_sampling_param reads only
{repetition_penalty, temperature, top_k, top_p, min_p, max_new_tokens} from
generation_config.json -- do_sample is ignored -- so greedy must be spelled out
as temperature 0.0 / top_p 1.0. top_k is dropped rather than set to -1, because
a -1 sentinel can make a later save_pretrained raise.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

SNAP = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d",
)
COPY = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
)


def finalize(ckpt: str, src: str = SNAP) -> dict:
    for f in COPY:
        s = os.path.join(src, f)
        if os.path.exists(s) and not os.path.exists(os.path.join(ckpt, f)):
            shutil.copy(s, os.path.join(ckpt, f))
    gc = json.load(open(os.path.join(src, "generation_config.json")))
    gc.pop("top_k", None)
    gc.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0})
    gc["eos_token_id"] = [1, 106]
    json.dump(gc, open(os.path.join(ckpt, "generation_config.json"), "w"), indent=2)
    # transformers 4.57 writes "dtype"; keep the legacy key too for older loaders
    cfgp = os.path.join(ckpt, "config.json")
    cfg = json.load(open(cfgp))
    if "torch_dtype" not in cfg:
        cfg["torch_dtype"] = cfg.get("dtype", "bfloat16")
        json.dump(cfg, open(cfgp, "w"), indent=2)
    missing = [f for f in ("config.json", "generation_config.json", "tokenizer.json") if not os.path.exists(os.path.join(ckpt, f))]
    assert not missing, missing
    return gc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--src", default=SNAP)
    a = ap.parse_args()
    print(json.dumps(finalize(a.ckpt, a.src), indent=2))
