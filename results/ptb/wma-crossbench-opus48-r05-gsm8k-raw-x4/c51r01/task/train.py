#!/usr/bin/env python3
import os, argparse, torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from trl import SFTConfig, SFTTrainer

SNAP = os.environ["PTB_BASE_MODEL_SNAPSHOT"]

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--data", nargs="+", required=True, help="jsonl files")
    p.add_argument("--repeat", nargs="+", type=int, default=None, help="repeat count per data file")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=-1)
    return p.parse_args()

def main():
    a = parse()
    tok = AutoTokenizer.from_pretrained(SNAP)
    with open("templates/gemma3.jinja") as f:
        tok.chat_template = f.read()

    dsets = []
    reps = a.repeat if a.repeat else [1]*len(a.data)
    for f, r in zip(a.data, reps):
        d = load_dataset("json", data_files=f, split="train")
        for _ in range(r):
            dsets.append(d)
    ds = concatenate_datasets(dsets).shuffle(seed=42)
    if a.limit:
        ds = ds.select(range(a.limit))
    print("Total training examples:", len(ds))

    model = Gemma3ForConditionalGeneration.from_pretrained(
        SNAP, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    n_train = 0
    for n, p in model.named_parameters():
        if "vision_tower" in n or "multi_modal_projector" in n:
            p.requires_grad = False
        elif p.requires_grad:
            n_train += p.numel()
    print(f"Trainable params: {n_train/1e9:.2f}B")

    cfg = SFTConfig(
        output_dir=a.out,
        per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=a.accum,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=3,
        max_length=a.max_len,
        packing=False,
        completion_only_loss=True,
        dataset_num_proc=8,
        report_to=[],
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    # sanity: show one tokenized example's label masking
    ex = trainer.train_dataset[0]
    print("keys:", list(ex.keys()))
    if "labels" in ex:
        lab = ex["labels"]
        print("n_labels:", len(lab), "n_supervised:", sum(1 for x in lab if x != -100))
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    # copy multimodal preprocessing configs so vLLM can load the model
    import shutil
    for fn in ["preprocessor_config.json", "processor_config.json", "generation_config.json"]:
        src = os.path.join(SNAP, fn)
        dst = os.path.join(a.out, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    print("Saved to", a.out)

if __name__ == "__main__":
    main()
