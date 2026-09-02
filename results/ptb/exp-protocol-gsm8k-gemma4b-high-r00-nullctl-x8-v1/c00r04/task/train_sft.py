#!/usr/bin/env python3
"""Full fine-tune of the text-only gemma-3-4b-pt decoder on GSM8K-style CoT data."""
import argparse, json, math, os, random, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, Gemma3ForCausalLM, Trainer,
                          TrainingArguments, set_seed)

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

END = "<end_of_turn>"


def build_fewshot_pool():
    """Few-shot exemplars rendered exactly like inspect_evals' sample_to_fewshot,
    drawn from the GSM8K *train* split."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    pool = []
    for q, a in zip(ds["question"], ds["answer"]):
        reasoning, _, target = a.partition("####")
        pool.append(f"{q}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {target.strip()}")
    return pool


class SFTData(Dataset):
    def __init__(self, records, tok, fewshot_pool, max_len, seed=0,
                 p_zero=0.55, p_ten=0.20):
        self.tok, self.max_len = tok, max_len
        self.examples = []
        rng = random.Random(seed)
        n_skip = 0
        prompts, comps = [], []
        for r in records:
            u = rng.random()
            if u < p_zero:
                k = 0
            elif u < p_zero + p_ten:
                k = 10
            else:
                k = rng.choice([1, 2, 3, 4, 5, 6, 8])
            msgs = []
            if k:
                shots = rng.sample(fewshot_pool, k)
                msgs.append({"role": "system", "content": "\n\n".join(shots)})
            msgs.append({"role": "user",
                         "content": MATH_PROMPT_TEMPLATE.format(prompt=r["question"])})
            prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True))
            comps.append(r["solution"].strip() + END)
        print("rendered prompts, tokenizing...", flush=True)
        B = 2000
        p_tok, c_tok = [], []
        for i in range(0, len(prompts), B):
            p_tok += tok(prompts[i:i + B], add_special_tokens=False)["input_ids"]
            c_tok += tok(comps[i:i + B], add_special_tokens=False)["input_ids"]
        for p_ids, c_ids in zip(p_tok, c_tok):
            if len(p_ids) + len(c_ids) > max_len:
                n_skip += 1
                continue
            self.examples.append((p_ids, c_ids))
        print(f"dataset: {len(self.examples)} examples ({n_skip} skipped for length)")
        self.lengths = [len(a) + len(b) for a, b in self.examples]
        print(f"total tokens: {sum(self.lengths)/1e6:.1f}M  mean {np.mean(self.lengths):.0f}"
              f"  p95 {np.percentile(self.lengths,95):.0f}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        p, c = self.examples[i]
        ids = p + c
        labels = [-100] * len(p) + list(c)
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        n = ((n + 15) // 16) * 16
        input_ids, labels, attn = [], [], []
        for f in feats:
            L = len(f["input_ids"])
            pad = n - L
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * L + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--extra-data", default=None,
                    help="comma separated extra jsonl files")
    ap.add_argument("--model", default="base_text")
    ap.add_argument("--out", default="runs/sft1")
    ap.add_argument("--n", type=int, default=0, help="subsample size (0=all)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-zero", type=float, default=0.55)
    ap.add_argument("--p-ten", type=float, default=0.20)
    ap.add_argument("--save-steps", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    recs = [json.loads(l) for l in open(args.data)]
    if args.extra_data:
        for p in args.extra_data.split(","):
            if p.strip():
                extra = [json.loads(l) for l in open(p.strip())]
                print(f"extra {p}: {len(extra)}")
                recs += extra
    rng = random.Random(args.seed)
    rng.shuffle(recs)
    if args.n:
        recs = recs[: args.n]
    print("records:", len(recs))

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open("templates/gemma3.jinja").read()

    ds = SFTData(recs, tok, build_fewshot_pool(), args.max_len, seed=args.seed,
                 p_zero=args.p_zero, p_ten=args.p_ten)

    from liger_kernel.transformers import apply_liger_kernel_to_gemma3_text
    apply_liger_kernel_to_gemma3_text(fused_linear_cross_entropy=True)
    model = Gemma3ForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="flash_attention_2")
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        adam_beta2=0.95,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        logging_steps=5,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        group_by_length=True,
        length_column_name="length",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()

    model.config.use_cache = True
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    # ship a generation config that stops on <end_of_turn>
    from transformers import GenerationConfig
    GenerationConfig(bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
                     cache_implementation="hybrid", do_sample=True,
                     top_k=64, top_p=0.95).save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
