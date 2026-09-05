#!/usr/bin/env python3
import argparse, json, os
import torch
from transformers import (AutoTokenizer, Gemma3ForCausalLM, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)
from datasets import Dataset

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_data.jsonl")
    ap.add_argument("--out", default="sft_out")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "right"

    # Build tokenized dataset with completion-only loss masking.
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]
    print("rows:", len(rows))

    TPL = open("templates/gemma3.jinja").read()

    def encode(ex):
        user = MATH_PROMPT_TEMPLATE.format(prompt=ex["question"])
        base_msgs = []
        if ex.get("system"):
            base_msgs.append({"role": "system", "content": ex["system"]})
        base_msgs.append({"role": "user", "content": user})
        msgs_full = base_msgs + [{"role": "assistant", "content": ex["reasoning_answer"]}]
        full = tok.apply_chat_template(msgs_full, tokenize=False,
                                       add_generation_prompt=False,
                                       chat_template=TPL)
        prompt = tok.apply_chat_template(base_msgs,
                                         tokenize=False, add_generation_prompt=True,
                                         chat_template=TPL)
        full_ids = tok(full, add_special_tokens=False)["input_ids"]
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        labels = list(full_ids)
        plen = min(len(prompt_ids), len(full_ids))
        for i in range(plen):
            labels[i] = -100
        return {"input_ids": full_ids, "labels": labels,
                "attention_mask": [1] * len(full_ids)}

    ds0 = Dataset.from_list(rows)
    ds = ds0.map(encode, remove_columns=ds0.column_names, num_proc=8)
    ds = ds.filter(lambda e: len(e["input_ids"]) <= args.maxlen, num_proc=8)
    # quick sanity on masking
    ex0 = ds[0]
    ntrained = sum(1 for x in ex0["labels"] if x != -100)
    print("example len", len(ex0["input_ids"]), "trained tokens", ntrained)

    model = Gemma3ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                              attn_implementation=args.attn)
    model.config.use_cache = False

    collator = DataCollatorForSeq2Seq(tok, model=model, padding=True,
                                      label_pad_token_id=-100)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        optim="adamw_torch_fused",
        group_by_length=True,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=collator)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    # copy generation config for correct eos tokens
    import shutil
    shutil.copy(os.path.join(BASE, "generation_config.json"),
                os.path.join(args.out, "generation_config.json"))
    print("saved to", args.out)


if __name__ == "__main__":
    main()
