"""Rewrite a checkpoint's generation_config.json.

vLLM logs "Default sampling parameters have been overridden by the model's
Hugging Face generation config" (see logs/exp-01.log), i.e. the server's default
sampling params come from this file. The base gemma-3-4b-pt snapshot ships
do_sample=true, top_k=64, top_p=0.95 and no temperature, so the harness decodes
by sampling unless the checkpoint says otherwise.
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["greedy", "base"], required=True)
    args = ap.parse_args()

    p = Path(args.model) / "generation_config.json"
    gc = json.loads(p.read_text())
    if args.mode == "greedy":
        gc["do_sample"] = False
        gc["temperature"] = 0.0
        gc.pop("top_k", None)
        gc.pop("top_p", None)
    else:
        gc["do_sample"] = True
        gc["top_k"] = 64
        gc["top_p"] = 0.95
        gc.pop("temperature", None)
    assert gc["eos_token_id"] == [1, 106], gc
    p.write_text(json.dumps(gc, indent=2) + "\n")
    print(f"{p}: {gc}")


if __name__ == "__main__":
    main()
