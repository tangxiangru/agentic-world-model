"""Copy a chosen checkpoint into final_model/ as real files, set greedy decoding,
and verify the result loads the way the grader will load it.

Guards the two packaging pitfalls: a final_model/ that vLLM cannot load, and a
generation_config that silently reverts to the base snapshot's sampling params
(train_sft.py's finalize() copies the base file into every checkpoint).
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_SNAPSHOT

ROOT = Path(__file__).resolve().parent.parent
SKIP = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
        "training_args.bin"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default=str(ROOT / "final_model"))
    ap.add_argument("--scoreboard", default=str(ROOT / "eval" / "scoreboard.json"),
                    help="JSON {checkpoint_path: accuracy} of the n=1319 reads")
    ap.add_argument("--allow-not-best", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    # guard: never overwrite a packaged winner with a worse candidate
    sb_path = Path(args.scoreboard)
    if sb_path.exists():
        sb = json.loads(sb_path.read_text())
        if str(src) not in sb:
            raise SystemExit(f"{src} has no recorded n=1319 score in {sb_path}; refusing")
        best = max(sb, key=lambda k: sb[k])
        if sb[str(src)] < sb[best] and not args.allow_not_best:
            raise SystemExit(
                f"refusing: {src} scored {sb[str(src)]} but {best} scored {sb[best]}")
        print(f"[guard] {src} is the best recorded candidate ({sb[str(src)]})")
    elif not args.allow_not_best:
        raise SystemExit(f"no scoreboard at {sb_path}; refusing to package blind")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.name in SKIP or f.is_dir():
            continue
        shutil.copyfile(f, dst / f.name)      # resolves symlinks into real files
        print(f"  {f.name} ({(dst / f.name).stat().st_size / 1e6:.1f} MB)")

    # tokenizer files the checkpoint may not carry
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "added_tokens.json", "tokenizer.model", "preprocessor_config.json",
                 "processor_config.json"):
        if not (dst / name).exists() and (Path(BASE_SNAPSHOT) / name).exists():
            shutil.copyfile(Path(BASE_SNAPSHOT) / name, dst / name)
            print(f"  {name} (from base snapshot)")

    subprocess.run([sys.executable, str(ROOT / "scripts" / "set_decode.py"),
                    "--model", str(dst), "--mode", "greedy"], check=True)

    gc = json.loads((dst / "generation_config.json").read_text())
    cfg = json.loads((dst / "config.json").read_text())
    assert gc["eos_token_id"] == [1, 106], gc
    assert gc.get("temperature") == 0.0 and gc.get("do_sample") is False, gc
    assert "gemma" in cfg["architectures"][0].lower(), cfg["architectures"]
    print(f"[ok] architectures={cfg['architectures']}  generation_config={gc}")

    # the grader loads final_model/ from a fresh process; make sure that works
    from transformers import AutoConfig, AutoTokenizer
    AutoConfig.from_pretrained(str(dst))
    tok = AutoTokenizer.from_pretrained(str(dst))
    print(f"[ok] tokenizer loads, <end_of_turn>={tok.convert_tokens_to_ids('<end_of_turn>')}")
    idx = json.loads((dst / "model.safetensors.index.json").read_text())
    shards = set(idx["weight_map"].values())
    missing = [s for s in shards if not (dst / s).exists()]
    assert not missing, missing
    print(f"[ok] all {len(shards)} weight shards present; final_model at {dst}")


if __name__ == "__main__":
    main()
