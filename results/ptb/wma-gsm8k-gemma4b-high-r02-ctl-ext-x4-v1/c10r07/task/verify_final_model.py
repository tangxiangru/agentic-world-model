#!/usr/bin/env python3
"""Pre-deadline check on final_model/, against the final_model_not_loadable pitfall.

The grader loads final_model/ with vLLM from a fresh process, using
templates/gemma3.jinja and match(numeric=True, location="end"). This checks, on
CPU only, everything that can be checked without a GPU:

  1. no symlinks and no adapter-only directory: real weight files, real bytes;
  2. config.json architecture is one evaluate.py's model_type() resolves, and it
     resolves to the gemma template;
  3. the tokenizer loads from the directory and round-trips <end_of_turn>;
  4. generation_config.json still yields temperature 0.0 in to_diff_dict(),
     which is the dict vLLM reads to set its default sampling params;
  5. the safetensors index names every shard present, and the shards open.
"""
from __future__ import annotations

import json
import os
import sys

FINAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_model")


def fail(msg: str) -> None:
    print(f"FAIL {msg}")
    sys.exit(1)


def main() -> None:
    ok = []

    if not os.path.isdir(FINAL):
        fail("final_model/ does not exist")

    links = [f for f in os.listdir(FINAL) if os.path.islink(os.path.join(FINAL, f))]
    if links:
        fail(f"final_model/ contains symlinks, which will not survive being moved: {links}")
    ok.append("no symlinks")

    if any(f.startswith("adapter_") for f in os.listdir(FINAL)):
        fail("final_model/ looks like a LoRA adapter directory; merge the adapter first")
    ok.append("not an adapter dir")

    cfg = json.load(open(os.path.join(FINAL, "config.json")))
    arch = cfg["architectures"][0].lower()
    if "gemma" not in arch:
        fail(f"evaluate.py's model_type() would not resolve architecture {arch}")
    ok.append(f"architecture {cfg['architectures'][0]} -> gemma3.jinja")

    idx = json.load(open(os.path.join(FINAL, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    total = 0
    for s in shards:
        p = os.path.join(FINAL, s)
        if not os.path.isfile(p):
            fail(f"index names {s} but it is missing")
        total += os.path.getsize(p)
    if total < 5e9:
        fail(f"weights total only {total/2**30:.1f} GiB; a bf16 4B model should be ~8 GiB")
    ok.append(f"{len(shards)} shard(s), {total/2**30:.1f} GiB")

    from safetensors import safe_open
    with safe_open(os.path.join(FINAL, shards[0]), framework="pt") as f:
        _ = next(iter(f.keys()))
    ok.append("safetensors open")

    from transformers import AutoTokenizer, GenerationConfig
    tok = AutoTokenizer.from_pretrained(FINAL)
    if tok.convert_tokens_to_ids("<end_of_turn>") != 106:
        fail("tokenizer does not map <end_of_turn> to 106")
    ok.append("tokenizer loads, <end_of_turn>=106")

    g = GenerationConfig.from_pretrained(FINAL)
    diff = g.to_diff_dict()
    if diff.get("temperature") != 0.0:
        fail(f"generation_config would not give vLLM greedy decoding: {diff}")
    eos = diff.get("eos_token_id") or g.eos_token_id
    if not (isinstance(eos, list) and 106 in eos):
        fail(f"eos_token_id does not include 106 <end_of_turn>: {eos}")
    ok.append(f"greedy decode + eos {eos}")

    # the grader renders with templates/gemma3.jinja, not with anything in the model dir
    import common
    p = common.render_prompt(tok, "A shop sells 3 pens at $4 each. Total cost?", True)
    if not p.endswith("<start_of_turn>model\n"):
        fail("the grader's template does not render as expected against this tokenizer")
    ok.append(f"grader template renders (sha {common.chat_template_sha()})")

    for line in ok:
        print("PASS", line)
    print("final_model/ is loadable and correctly configured")


if __name__ == "__main__":
    main()
