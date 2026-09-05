"""Rewrite a checkpoint's generation_config.json.

The grader sends only max_tokens; inspect's vLLM provider passes no sampling
parameters, and vLLM's default generation_config="auto" fills the rest from the
checkpoint's own generation_config.json (ModelConfig.get_diff_sampling_param
reads temperature / top_p / top_k / min_p / repetition_penalty from it). The
gemma-3-4b-pt snapshot ships do_sample=true, top_k=64, top_p=0.95 and no
temperature, so the benchmark samples at temperature 1.0 unless the checkpoint
says otherwise. do_sample is NOT one of the keys vLLM reads - only temperature
switches it to greedy.
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["greedy", "inherit"], default="greedy")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override the temperature written for --mode greedy")
    args = ap.parse_args()

    path = os.path.join(args.model, "generation_config.json")
    with open(path) as f:
        cfg = json.load(f)
    print("before:", json.dumps(cfg))

    if args.mode == "greedy":
        cfg["do_sample"] = False
        cfg["temperature"] = 0.0 if args.temperature is None else args.temperature
        cfg["top_p"] = 1.0
        cfg["top_k"] = 0
    else:
        cfg.pop("temperature", None)
        cfg["do_sample"] = True
        cfg["top_p"] = 0.95
        cfg["top_k"] = 64

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("after: ", json.dumps(cfg))


if __name__ == "__main__":
    main()
