#!/usr/bin/env python3
"""CPU-only: materialize checked rendered token artifact. No model construction."""
import os, sys, argparse
os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
from pathlib import Path
from transformers import AutoTokenizer
from awm.exp_protocol.rendered_training import RenderedSettings, RenderedTrainingBundle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import render

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TASK = Path("/home/ben/task")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)      # jsonl of {prompt, completion}
    ap.add_argument("--output", required=True)       # token artifact dir
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAP, local_files_only=True, token=False)
    settings = RenderedSettings(
        mode="separate_concat",
        prompt_mode="pre_rendered",
        max_seq_len=args.max_seq_len,
        stop_token="<end_of_turn>",
        answer_marker="ANSWER: ",
        tail_text="",
        pad_to_multiple_of=8,
        seed=0,
        limit=args.limit,
    )
    prepared = RenderedTrainingBundle.prepare(
        sources=[Path(args.source)],
        render=render,
        tokenizer=tok,
        template_bytes=(TASK / "templates/gemma3.jinja").read_bytes(),
        settings=settings,
        source_files=[TASK / "scripts/preprocessing.py",
                      TASK / "scripts/prepare_tokens.py",
                      TASK / "scripts/train_sft.py"],
        output=Path(args.output),
    )
    print("DECLARATION:", prepared.declaration)
    print("DATA_ENTRY:", prepared.data_entry)
    print("REPORT:", prepared.report)

if __name__ == "__main__":
    main()
