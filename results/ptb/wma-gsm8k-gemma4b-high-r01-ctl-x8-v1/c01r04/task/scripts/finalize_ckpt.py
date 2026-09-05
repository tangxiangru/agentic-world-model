#!/usr/bin/env python3
"""Make a trainer checkpoint directory loadable by the grader's vLLM, in place or into a copy.

Trainer.save_model writes weights + config.json + generation_config.json; the tokenizer we
add ourselves.  Gemma3 is a multimodal architecture, so vLLM also wants the processor files
that came with the snapshot (preprocessor_config.json, processor_config.json,
added_tokens.json, tokenizer.model).  Missing those is the `final_model_not_loadable` pitfall.

Optionally rewrites generation_config.json: the snapshot ships do_sample/top_k/top_p and no
temperature, and vLLM's `--generation-config auto` (the default) adopts those as the server's
default sampling params, so the graded run samples at T=1.0 unless we say otherwise.
"""
import argparse, json, os, shutil, sys

AUX = ["preprocessor_config.json", "processor_config.json", "added_tokens.json",
       "tokenizer.model", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dest", default=None, help="copy to this dir instead of editing in place")
    ap.add_argument("--snapshot", default=os.environ["PTB_BASE_MODEL_SNAPSHOT"])
    ap.add_argument("--decode", choices=["inherit", "greedy"], default="inherit")
    ap.add_argument("--verify", action="store_true", help="load the result on CPU with transformers")
    a = ap.parse_args()

    dest = a.dest or a.ckpt
    if a.dest:
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(a.ckpt, dest, ignore=shutil.ignore_patterns(
            "optimizer.pt", "scheduler.pt", "rng_state*", "trainer_state.json", "training_args.bin"))

    for fn in AUX:
        src, dst = os.path.join(a.snapshot, fn), os.path.join(dest, fn)
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy(os.path.realpath(src), dst)
            print("copied", fn)

    gc_path = os.path.join(dest, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
    if a.decode == "greedy":
        gc.pop("top_k", None)
        gc.pop("top_p", None)
        gc["do_sample"] = False
        gc["temperature"] = 0.0
    gc["eos_token_id"] = [1, 106]
    gc["bos_token_id"] = 2
    gc["pad_token_id"] = 0
    json.dump(gc, open(gc_path, "w"), indent=2)
    print("generation_config:", json.dumps(gc))

    missing = [f for f in ["config.json", "generation_config.json", "tokenizer_config.json"]
               if not os.path.exists(os.path.join(dest, f))]
    assert not missing, f"missing {missing}"
    print("ok", dest)

    if a.verify:
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
        tok = AutoTokenizer.from_pretrained(dest)
        AutoProcessor.from_pretrained(dest)
        m = AutoModelForImageTextToText.from_pretrained(dest, dtype=torch.bfloat16,
                                                        device_map="cpu")
        n = sum(p.numel() for p in m.parameters())
        print(f"CPU load OK: {n/1e9:.2f}B params, tokenizer vocab {len(tok)}")


if __name__ == "__main__":
    main()
