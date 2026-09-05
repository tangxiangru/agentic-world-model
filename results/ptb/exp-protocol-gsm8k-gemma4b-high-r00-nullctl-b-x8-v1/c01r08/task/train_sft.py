#!/usr/bin/env python3
import argparse, json, os, random, math
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments, set_seed)

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

PROMPT = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
    "{q}\n\n"
    'Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) '
    "where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.\n\n"
    "Reasoning:"
)


def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE)
    with open("templates/gemma3.jinja") as f:
        tok.chat_template = f.read()
    return tok


class SFTData(Dataset):
    def __init__(self, path, tok, max_len=1024, limit=None, fewshot_frac=0.0,
                 fewshot_pool=None, seed=0):
        self.ex = []
        rng = random.Random(seed)
        rows = [json.loads(l) for l in open(path)]
        if limit:
            rows = rows[:limit]
        bos = tok.bos_token
        eot = "<end_of_turn>"
        nskip = 0
        for r in rows:
            user = PROMPT.format(q=r["question"])
            if fewshot_pool and rng.random() < fewshot_frac:
                k = rng.choice([1, 2, 3, 4])
                shots = rng.sample(fewshot_pool, k)
                user = "\n\n".join(shots) + "\n\n" + user
            prompt = f"{bos}<start_of_turn>user\n{user.strip()}{eot}\n<start_of_turn>model\n"
            comp = r["solution"].strip() + eot + "\n"
            pi = tok(prompt, add_special_tokens=False)["input_ids"]
            ci = tok(comp, add_special_tokens=False)["input_ids"]
            if len(pi) + len(ci) > max_len:
                nskip += 1
                continue
            ids = pi + ci
            labels = [-100] * len(pi) + ci[:]
            self.ex.append((ids, labels))
        print(f"dataset: {len(self.ex)} examples, skipped {nskip} too long")
        self.lens = [len(a) for a, _ in self.ex]
        print("token stats: mean %.1f p95 %d max %d total %.2fM" % (
            np.mean(self.lens), np.percentile(self.lens, 95), max(self.lens),
            sum(self.lens) / 1e6))

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        ids, lab = self.ex[i]
        return {"input_ids": ids, "labels": lab, "length": len(ids)}


class Collator:
    def __init__(self, pad_id):
        self.pad = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        m = int(math.ceil(m / 8) * 8)
        ii, ll, am = [], [], []
        for f in feats:
            n = m - len(f["input_ids"])
            ii.append(f["input_ids"] + [self.pad] * n)
            ll.append(f["labels"] + [-100] * n)
            am.append([1] * len(f["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
                "attention_mask": torch.tensor(am)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_v1.jsonl")
    ap.add_argument("--out", default="runs/sft_v1")
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--fewshot-frac", type=float, default=0.0)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--save-steps", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = build_tokenizer()

    fewshot_pool = None
    if args.fewshot_frac > 0:
        from datasets import load_dataset
        import re as _re
        d = load_dataset("openai/gsm8k", "main")["train"]
        fewshot_pool = []
        for r in list(d)[:3000]:
            sol, tgt = r["answer"].split("####")
            fewshot_pool.append(
                f"{r['question']}\n\nReasoning:\n{sol.strip()}\n\nANSWER: {tgt.strip()}")

    ds = SFTData(args.data, tok, args.max_len, args.limit,
                 args.fewshot_frac, fewshot_pool, args.seed)

    try:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3
        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("liger kernel applied")
    except Exception as e:
        print("liger failed:", e)

    model = AutoModelForCausalLM.from_pretrained(
        args.init, dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    nfroze = 0
    for n, p in model.named_parameters():
        if "vision_tower" in n or "multi_modal_projector" in n:
            p.requires_grad_(False)
            nfroze += p.numel()
    print("frozen params: %.1fM" % (nfroze / 1e6))
    print("trainable: %.1fM" % (sum(p.numel() for p in model.parameters()
                                    if p.requires_grad) / 1e6))

    ta = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio},
        warmup_steps=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        optim=args.optim,
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=ta, train_dataset=ds,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.config.use_cache = True
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    import shutil
    for f in ["preprocessor_config.json", "processor_config.json", "generation_config.json"]:
        src = os.path.join(BASE, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    print("saved to", args.out)


if __name__ == "__main__":
    main()
