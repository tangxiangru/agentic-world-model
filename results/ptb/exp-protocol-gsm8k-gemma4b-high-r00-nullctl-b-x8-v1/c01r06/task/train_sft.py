"""Full-parameter SFT of gemma-3-4b-pt on math CoT, completion-only loss."""
import argparse
import json
import os
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (AutoTokenizer, Gemma3ForConditionalGeneration,
                          Trainer, TrainingArguments)


class SparseHeadTrainer(Trainer):
    """Apply the (262k-wide) LM head only at supervised positions.

    Gemma-3's vocabulary makes full-sequence logits the dominant cost in both
    memory and FLOPs; only ~half the positions carry a label here, and the
    padding never does.
    """

    LOGIT_CHUNK = 16384

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        core = model.model
        hidden = core(input_ids=inputs["input_ids"],
                      attention_mask=inputs["attention_mask"]).last_hidden_state
        tgt = labels[:, 1:]
        sel = tgt != -100
        h = hidden[:, :-1, :][sel]
        t = tgt[sel]
        total = 0.0
        for i in range(0, t.numel(), self.LOGIT_CHUNK):
            logits = model.lm_head(h[i: i + self.LOGIT_CHUNK]).float()
            total = total + F.cross_entropy(logits, t[i: i + self.LOGIT_CHUNK],
                                            reduction="sum")
        denom = num_items_in_batch if num_items_in_batch is not None else t.numel()
        if torch.is_tensor(denom):
            denom = denom.to(total.device)
        return total / denom

from common import fewshot_block, render_prompt, split_gsm8k_answer

SNAP = os.environ["PTB_BASE_MODEL_SNAPSHOT"]
EOT = 106


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sft_v1.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--init", default=SNAP)
    p.add_argument("--n", type=int, default=-1)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--micro-bs", type=int, default=4)
    p.add_argument("--accum", type=int, default=16)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--fewshot-frac", type=float, default=0.15)
    p.add_argument("--save-steps", type=int, default=400)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn", default="flash_attention_2")
    p.add_argument("--max-steps", type=int, default=-1)
    return p.parse_args()


def build_fewshot_pool(tok):
    """Few-shot demos drawn from the GSM8K *train* split, same shape as the eval's."""
    from datasets import load_dataset
    g = load_dataset("openai/gsm8k", "main", split="train")
    pool = []
    for q, a in zip(g["question"], g["answer"]):
        reasoning, target = split_gsm8k_answer(a)
        pool.append(fewshot_block(q, reasoning, target))
    return pool


def main():
    args = parse()
    random.seed(args.seed)
    np.random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(SNAP)

    rows = [json.loads(l) for l in open(args.data)]
    random.shuffle(rows)
    if args.n > 0:
        rows = rows[: args.n]
    print(f"{len(rows)} training rows")

    fs_pool = build_fewshot_pool(tok) if args.fewshot_frac > 0 else []

    rng = random.Random(args.seed + 1)
    prompts, completions = [], []
    for r in rows:
        prefix = ""
        if fs_pool and rng.random() < args.fewshot_frac:
            k = rng.randint(1, 10)
            prefix = "\n\n".join(rng.sample(fs_pool, k))
        prompts.append("<bos>" + render_prompt(r["question"], prefix))
        completions.append(r["solution"])

    ds = Dataset.from_dict({"prompt": prompts, "completion": completions})

    def tokenize(batch):
        p_ids = tok(batch["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = tok(batch["completion"], add_special_tokens=False)["input_ids"]
        out_ids, out_lab, out_len = [], [], []
        for a, b in zip(p_ids, c_ids):
            b = b + [EOT]
            ids = a + b
            if len(ids) > args.max_len:
                # Drop few-shot demos from the front, but keep the
                # "<bos><start_of_turn>user\n" header intact.
                over = len(ids) - args.max_len
                if over < len(a) - 64:
                    a = a[:4] + a[4 + over:]
                    ids = a + b
                else:
                    out_ids.append(None); out_lab.append(None); out_len.append(0)
                    continue
            out_ids.append(ids)
            out_lab.append([-100] * len(a) + b)
            out_len.append(len(ids))
        return {"input_ids": out_ids, "labels": out_lab, "length": out_len}

    ds = ds.map(tokenize, batched=True, batch_size=1000, num_proc=16,
                remove_columns=["prompt", "completion"])
    ds = ds.filter(lambda x: x["length"] > 0, num_proc=16)
    lens = np.array(ds["length"])
    print(f"kept {len(ds)}  tokens={lens.sum()/1e6:.1f}M  mean={lens.mean():.0f} "
          f"p95={np.percentile(lens,95):.0f} max={lens.max()}")

    def collate(feats):
        m = max(len(f["input_ids"]) for f in feats)
        m = (m + 7) // 8 * 8
        ids = torch.full((len(feats), m), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(feats), m), -100, dtype=torch.long)
        att = torch.zeros((len(feats), m), dtype=torch.long)
        for i, f in enumerate(feats):
            n = len(f["input_ids"])
            ids[i, :n] = torch.tensor(f["input_ids"])
            lab[i, :n] = torch.tensor(f["labels"])
            att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    print("loading model ...")
    try:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.init, dtype=torch.bfloat16, attn_implementation=args.attn)
    except Exception as e:
        print("attn impl", args.attn, "failed:", e, "-> eager")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.init, dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    # Train the language model only; the vision tower is irrelevant to GSM8K.
    for n_, p_ in model.named_parameters():
        if n_.startswith("model.vision_tower") or n_.startswith("model.multi_modal_projector"):
            p_.requires_grad_(False)
    # Trainer validates generation_config on every save; keep a strictly-valid one
    # here and apply the greedy inference defaults afterwards via set_gencfg.py.
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid", do_sample=True, top_k=64, top_p=0.95)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e9:.2f}B")

    targs = TrainingArguments(
        output_dir=args.out,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.micro_bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio},
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        max_grad_norm=1.0,
        adam_beta2=0.95,
        optim="adamw_torch_fused",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=1,
        save_safetensors=True,
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=4,
        report_to=[],
        seed=args.seed,
    )

    trainer = SparseHeadTrainer(model=model, args=targs, train_dataset=ds,
                                data_collator=collate)
    trainer.train()
    print("peak mem GB:", torch.cuda.max_memory_allocated() / 1e9)

    print("saving ->", args.out)
    model.config.use_cache = True
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    for fn in ["preprocessor_config.json", "processor_config.json",
               "added_tokens.json", "special_tokens_map.json", "tokenizer.model"]:
        src = os.path.join(SNAP, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, fn))
    # keep generation defaults (eos = <eos>,<end_of_turn>) so vLLM stops correctly
    shutil.copy(os.path.join(SNAP, "generation_config.json"),
                os.path.join(args.out, "generation_config.json"))
    print("done")


if __name__ == "__main__":
    main()
