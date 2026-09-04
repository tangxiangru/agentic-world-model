"""Uniform weight average (model soup) of checkpoints from one training run.

Averaging is done in float32 and cast back to bfloat16 once, so the rounding of
the individual bf16 checkpoints is not compounded per term.
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, GenerationConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_SNAPSHOT, load_tokenizer
from train_sft import finalize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[soup] averaging {len(args.ckpts)} checkpoints uniformly")
    model = AutoModelForCausalLM.from_pretrained(args.ckpts[0], dtype=torch.float32)
    acc = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for p in args.ckpts[1:]:
        other = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32)
        sd = other.state_dict()
        assert set(sd) == set(acc), "checkpoints have different parameter sets"
        for k in acc:
            acc[k] += sd[k]
        del other
        print(f"  added {p}")

    n = len(args.ckpts)
    for k in acc:
        acc[k] /= n
    model.load_state_dict(acc)
    model = model.to(torch.bfloat16)

    # the ingredients carry the greedy generation_config, which transformers'
    # save-time validator rejects (temperature with do_sample=False); save the
    # base config and let set_decode.py write the greedy one afterwards
    model.generation_config = GenerationConfig.from_pretrained(BASE_SNAPSHOT)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    load_tokenizer(BASE_SNAPSHOT).save_pretrained(args.out)
    finalize(args.out)
    print(f"[soup] wrote {args.out}")


if __name__ == "__main__":
    main()
