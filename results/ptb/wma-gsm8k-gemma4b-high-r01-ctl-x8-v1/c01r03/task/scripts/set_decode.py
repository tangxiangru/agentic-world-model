#!/usr/bin/env python3
"""Set the decoding defaults that vLLM will pick up for a checkpoint.

evaluate.py runs `vllm serve <dir>` with no sampling flags, and inspect-ai
sends no temperature, so vLLM falls back to ModelConfig.get_diff_sampling_param(),
which reads generation_config.json in the model directory. Only these keys are
read (vllm/config/model.py):

    repetition_penalty, temperature, top_k, top_p, min_p, max_new_tokens

`do_sample` is NOT among them, which is why the stock gemma-3-4b-pt config
(do_sample=true, top_k=64, top_p=0.95, no temperature) makes the grader sample
at temperature 1.0. Writing temperature/top_k/top_p explicitly is the only way
to change that from inside the model directory.

eos_token_id must keep both 1 (<eos>) and 106 (<end_of_turn>): 106 is the token
our SFT targets end with.
"""

from __future__ import annotations

import argparse
import json
import os

GREEDY = {"temperature": 0.0, "top_k": -1, "top_p": 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    ap.add_argument("--temperature", type=float, default=None)
    args = ap.parse_args()

    path = os.path.join(args.dir, "generation_config.json")
    cfg = json.load(open(path))
    before = dict(cfg)

    if args.mode == "greedy":
        cfg.update(GREEDY)
        cfg["do_sample"] = False
    else:
        cfg.update({"temperature": args.temperature or 1.0, "top_k": 64, "top_p": 0.95})
        cfg["do_sample"] = True

    eos = cfg.get("eos_token_id")
    assert eos == [1, 106] or eos == 106, f"unexpected eos_token_id {eos}"

    json.dump(cfg, open(path, "w"), indent=2)
    print("before:", json.dumps(before))
    print("after :", json.dumps(cfg))
    print("wrote", path)


if __name__ == "__main__":
    main()
