import argparse, os, json, torch
from transformers import (AutoTokenizer, AutoModelForImageTextToText,
                          Trainer, TrainingArguments, DataCollatorForSeq2Seq)
from datasets import load_dataset

def parse():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default="/home/ben/task/templates/gemma3_tool_calling.jinja")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--full_ft", action="store_true")
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--save_steps", type=int, default=0)
    return ap.parse_args()

def main():
    a=parse()
    tok=AutoTokenizer.from_pretrained(a.model)
    tok.chat_template=open(a.template).read()
    ds=load_dataset("json", data_files=a.data, split="train")
    if a.limit: ds=ds.select(range(min(a.limit,len(ds))))

    def tokenize(r):
        prompt_ids=tok(r["prompt"], add_special_tokens=False)["input_ids"]
        full_ids=tok(r["prompt"]+r["completion"], add_special_tokens=False)["input_ids"]
        n=len(prompt_ids)
        if full_ids[:n]!=prompt_ids:  # token seam mismatch -> use common prefix
            m=0
            for x,y in zip(full_ids,prompt_ids):
                if x==y: m+=1
                else: break
            n=m
        labels=[-100]*n + full_ids[n:]
        return {"input_ids":full_ids,"labels":labels,"attention_mask":[1]*len(full_ids)}

    ds=ds.map(tokenize, remove_columns=ds.column_names, num_proc=8)
    before=len(ds)
    ds=ds.filter(lambda r: len(r["input_ids"])<=a.max_seq_len)
    print(f"train rows {len(ds)} (dropped {before-len(ds)} over max_seq_len)")

    model=AutoModelForImageTextToText.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache=False

    if not a.full_ft:
        from peft import LoraConfig, get_peft_model
        targets=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        lc=LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                      bias="none", task_type="CAUSAL_LM", target_modules=targets)
        model=get_peft_model(model, lc)
        model.print_trainable_parameters()
    else:
        # freeze vision tower + multimodal projector; train language model only
        for n,p in model.named_parameters():
            if ("vision_tower" in n) or ("multi_modal_projector" in n):
                p.requires_grad=False

    collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100,
                                    return_tensors="pt")
    steps_arg={}
    if a.save_steps>0:
        steps_arg=dict(save_strategy="steps", save_steps=a.save_steps, save_total_limit=3)
    else:
        steps_arg=dict(save_strategy="no")
    targs=TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr,
        num_train_epochs=a.epochs, lr_scheduler_type=a.scheduler,
        warmup_ratio=a.warmup, weight_decay=a.wd, bf16=True,
        logging_steps=20, gradient_checkpointing=(not a.no_grad_ckpt),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[], seed=a.seed, dataloader_num_workers=4,
        remove_unused_columns=False, **steps_arg)
    trainer=Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    # save merged full model + tokenizer for vLLM
    merged_dir=os.path.join(a.out,"merged")
    os.makedirs(merged_dir, exist_ok=True)
    if not a.full_ft:
        model=model.merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    # loss log
    with open(os.path.join(a.out,"trainlog.json"),"w") as f:
        json.dump(trainer.state.log_history, f)
    print("SAVED_MERGED", merged_dir)

if __name__=="__main__":
    main()
