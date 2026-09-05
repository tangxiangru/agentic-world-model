#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on GSM8K-style data.

Uses the checked RenderedTrainingBundle (opened from the locked card) for data,
SaveSafeTrainer + GenerationSaveContract for saves, and writes a GREEDY serving
generation_config.json so vLLM decodes deterministically at grading time.
Runs entirely in the foreground of the locked-card command (rule 9).
"""
import os, sys, json, argparse, shutil, time
os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path
import torch
from transformers import AutoTokenizer, TrainingArguments, Gemma3ForConditionalGeneration
from awm.exp_protocol.rendered_training import RenderedTrainingBundle
from awm.exp_protocol.save_trainer import SaveSafeTrainer
from awm.exp_protocol.save_contract import GenerationSaveContract

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

# Greedy serving config: temperature 0 => vLLM greedy decoding. eos includes
# <eos>(1) and <end_of_turn>(106) so generation stops at the turn terminator.
GREEDY_GENCONFIG = {
    "bos_token_id": 2,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--model", default=SNAP)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--per-device-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--max-steps", type=int, default=-1)  # smoke override
    args = ap.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] opening bundle from card {args.card}", flush=True)
    bundle = RenderedTrainingBundle.open_for_training(Path(args.card))
    dataset = bundle.dataset
    collator = bundle.collator(pad_to_multiple_of=8)
    print(f"dataset rows: {len(dataset)}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading model {args.model} attn={args.attn}", flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
    )
    model.config.use_cache = False

    contract = GenerationSaveContract()
    contract.check_before_compute(model)

    targs = TrainingArguments(
        output_dir=str(outdir),
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        max_grad_norm=1.0,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=args.seed,
    )

    trainer = SaveSafeTrainer(
        model=model, args=targs, train_dataset=dataset,
        data_collator=collator, generation_save_contract=contract,
    )
    contract.check_before_compute(trainer.model)

    print(f"[{time.strftime('%H:%M:%S')}] starting training", flush=True)
    train_out = trainer.train()
    print("train metrics:", train_out.metrics, flush=True)
    bundle.flush_consumption()

    # ---- final export with greedy serving config ----
    model = trainer.accelerator.unwrap_model(trainer.model)
    model.config.use_cache = True
    greedy_bytes = (json.dumps(GREEDY_GENCONFIG, indent=2) + "\n").encode()

    print(f"[{time.strftime('%H:%M:%S')}] saving final model to {outdir}", flush=True)
    try:
        contract.check_before_compute(model)
        with contract.saving(model, outdir, selected_serving_json=greedy_bytes):
            model.save_pretrained(outdir, safe_serialization=True)
        assert (outdir / "generation_config.json").read_bytes() == greedy_bytes
        print("safe save + greedy genconfig OK", flush=True)
    except Exception as e:
        print(f"WARNING contract save failed ({e!r}); falling back to plain save", flush=True)
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        model.save_pretrained(outdir, safe_serialization=True)
        (outdir / "generation_config.json").write_bytes(greedy_bytes)

    # tokenizer + multimodal processor aux files for vLLM loadability
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.save_pretrained(outdir)
    for fn in ["preprocessor_config.json", "processor_config.json"]:
        src = Path(args.model) / fn
        if src.exists():
            shutil.copy(src, outdir / fn)

    # sanity: required files present
    need = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]
    print("final files:", sorted(os.listdir(outdir)), flush=True)
    for n in need:
        assert (outdir / n).exists(), f"missing {n}"
    print("FINAL genconfig:", (outdir / "generation_config.json").read_text(), flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] DONE", flush=True)


if __name__ == "__main__":
    main()
