#!/usr/bin/env python3
"""LoRA SFT of gemma-3-4b-pt on GSM8K-train (eval-format), then merge + save full model.

Targets end with <end_of_turn> (id 106, an eos) so vLLM stops cleanly.
Loss is completion-only (prompt tokens masked with -100).
Rendering uses the exact grader template templates/gemma3.jinja.
"""
import os, json, argparse, hashlib
import torch
from transformers import (AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)          # dir for merged full model
    p.add_argument("--adapter-out", default=None)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--template", default="templates/gemma3.jinja")
    p.add_argument("--warmup", type=float, default=0.03)
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    chat_template = open(args.template).read()
    tok = AutoTokenizer.from_pretrained(args.model)
    EOT = tok.convert_tokens_to_ids("<end_of_turn>")
    assert EOT == 106, EOT

    # ---- build examples ----
    rows = [json.loads(l) for l in open(args.data)]
    examples = []
    n_trunc = 0
    for r in rows:
        pre = ([{"role": "system", "content": r["system"]}] if r.get("system") else [])
        prompt_ids = tok.apply_chat_template(
            pre + [{"role": "user", "content": r["prompt"]}],
            tokenize=True, add_generation_prompt=True, chat_template=chat_template)
        full_ids = tok.apply_chat_template(
            pre + [{"role": "user", "content": r["prompt"]},
             {"role": "assistant", "content": r["completion"]}],
            tokenize=True, add_generation_prompt=False, chat_template=chat_template)
        assert full_ids[:len(prompt_ids)] == prompt_ids, "prompt not a prefix of full"
        # template appends '<end_of_turn>\n'; trim trailing newline so target ends at EOT (106)
        if len(full_ids) >= 2 and full_ids[-1] == 107 and full_ids[-2] == EOT:
            full_ids = full_ids[:-1]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        if len(full_ids) > args.max_seq_len:
            # LEFT-truncate: keep BOS + the last (max_seq_len-1) tokens, so the
            # target completion (and its EOT) at the tail is always preserved;
            # only the earliest fewshot context tokens are dropped.
            n_trunc += 1
            overflow = len(full_ids) - args.max_seq_len
            full_ids = full_ids[:1] + full_ids[1 + overflow:]
            labels = labels[:1] + labels[1 + overflow:]
            assert len(full_ids) == args.max_seq_len
            assert labels[-1] == full_ids[-1]  # completion tail intact
        examples.append({"input_ids": full_ids, "labels": labels,
                         "attention_mask": [1]*len(full_ids)})
    print(f"[data] {len(examples)} examples, truncated {n_trunc} "
          f"({100*n_trunc/len(examples):.2f}%)")

    # ---- rendered-training evidence: dump one example decoded ----
    ex0 = examples[0]
    sup = [t for t in ex0["labels"] if t != -100]
    with open("analysis/rendered_example.txt", "w") as f:
        f.write("FULL DECODED:\n"); f.write(tok.decode(ex0["input_ids"])); f.write("\n\n")
        f.write("SUPERVISED (labels != -100) DECODED:\n"); f.write(tok.decode(sup)); f.write("\n\n")
        f.write(f"last 5 supervised ids: {sup[-5:]} (EOT={EOT})\n")
        f.write(f"first prompt id: {ex0['input_ids'][0]} (bos={tok.bos_token_id})\n")
    assert sup[-1] == EOT, f"supervised must end with EOT, got {sup[-1]}"
    print(f"[render] supervised ends with EOT={EOT}: OK; last ids {sup[-3:]}")

    # ---- collator ----
    pad_id = tok.pad_token_id
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id]*n)
            labels.append(b["labels"] + [-100]*n)
            attn.append(b["attention_mask"] + [0]*n)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}

    # ---- model ----
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] trainable params: {n_train/1e6:.2f}M")
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.adapter_out or (args.out + "_adapter"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        seed=args.seed,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=examples,
                      data_collator=collate)
    train_out = trainer.train()
    print("[train] done", train_out.metrics)

    # ---- merge LoRA -> full model ----
    print("[merge] merging LoRA into base weights ...")
    merged = model.merge_and_unload()
    merged.config.use_cache = True

    os.makedirs(args.out, exist_ok=True)
    # keep model.generation_config valid (do_sample True) during native save
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    # save processor too (multimodal) for full loadability
    try:
        proc = AutoProcessor.from_pretrained(args.model)
        proc.save_pretrained(args.out)
    except Exception as e:
        print("[warn] processor save failed:", e)

    # ---- overwrite generation_config with greedy, valid decoder ----
    gen = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "cache_implementation": "hybrid",
        "do_sample": False,
    }
    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump(gen, f, indent=2)
    with open(os.path.join(args.out, "TRAIN_META.json"), "w") as f:
        json.dump({"final_loss": train_out.metrics.get("train_loss"),
                   "steps": train_out.metrics.get("step", None),
                   "metrics": train_out.metrics,
                   "args": vars(args)}, f, indent=2)
    print("[save] merged full model + tokenizer + processor ->", args.out)
    print("[save] files:", sorted(os.listdir(args.out)))

if __name__ == "__main__":
    main()
