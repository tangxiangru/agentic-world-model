"""Copy a checkpoint into final_model/ and prove it loads.

Guard: refuses unless --score is given and beats --incumbent, so a worse model
cannot be shipped over a better one by accident.

Also lets the decode contract be set explicitly: --greedy writes
temperature 0.0 / top_p 1.0 / top_k -1 into generation_config.json (vLLM reads
temperature, top_p, top_k, min_p, repetition_penalty and max_new_tokens from
that file), keeping eos_token_id [1, 106] so <end_of_turn> stays a stop token.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

NEEDED = ["config.json", "generation_config.json", "tokenizer.json",
          "tokenizer_config.json", "special_tokens_map.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--score", type=float, required=True)
    ap.add_argument("--incumbent", type=float, required=True)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.score <= args.incumbent and not args.force:
        sys.exit(f"REFUSING: candidate {args.score} does not beat incumbent {args.incumbent}")

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst,
                    ignore=shutil.ignore_patterns("checkpoint-*", "optimizer.pt",
                                                  "scheduler.pt", "rng_state*",
                                                  "trainer_state.json", "training_args.bin"))

    gcp = os.path.join(args.dst, "generation_config.json")
    gc = json.load(open(gcp))
    assert gc.get("eos_token_id") == [1, 106], f"eos_token_id lost: {gc.get('eos_token_id')}"
    if args.greedy:
        gc.update({"temperature": 0.0, "top_p": 1.0, "top_k": -1})
        gc.pop("do_sample", None)
        json.dump(gc, open(gcp, "w"), indent=2)
    print("generation_config:", json.dumps(json.load(open(gcp))))

    missing = [f for f in NEEDED if not os.path.exists(os.path.join(args.dst, f))]
    assert not missing, f"missing files in {args.dst}: {missing}"

    # Load once from a fresh CPU process, the way the grader loads it (fresh process).
    code = (
        "import os;os.environ['HF_HOME']='/home/ben/hf_cache';"
        "import transformers,json;"
        f"c=transformers.AutoConfig.from_pretrained({args.dst!r});"
        "cls=getattr(transformers,c.architectures[0]);"
        f"m=cls.from_pretrained({args.dst!r},dtype='bfloat16',device_map='cpu');"
        f"t=transformers.AutoTokenizer.from_pretrained({args.dst!r});"
        "print('LOAD_OK',c.architectures[0],sum(p.numel() for p in m.parameters())/1e9,'B',"
        "t.convert_tokens_to_ids('<end_of_turn>'))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr[-2000:])
    assert "LOAD_OK" in r.stdout, "final_model did not load in a fresh process"
    print(f"packaged {args.src} -> {args.dst}")


if __name__ == "__main__":
    main()
