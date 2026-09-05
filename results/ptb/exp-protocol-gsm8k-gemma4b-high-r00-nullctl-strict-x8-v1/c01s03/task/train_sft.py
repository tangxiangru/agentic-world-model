#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt (text tower) on math CoT data."""
import argparse, json, os, random, math
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoConfig, Gemma3ForCausalLM, Trainer,
                          TrainingArguments)

BASE = "base_model"
PROMPT = (
    'Solve the following math problem step by step. The last line of your response should be of the '
    'form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
    '{prompt}\n\n'
    'Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without '
    'quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed '
    'command.\n\nReasoning:'
)


def build_text(tok, question, solution, fewshots):
    """Return (full_ids, n_prompt_ids) using the eval chat template."""
    msgs = []
    if fewshots:
        msgs.append({"role": "system", "content": "\n\n".join(fewshots)})
    msgs.append({"role": "user", "content": PROMPT.format(prompt=question)})
    prompt_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full_str = prompt_str + solution + "<end_of_turn>\n"
    p_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]
    f_ids = tok(full_str, add_special_tokens=False)["input_ids"]
    return f_ids, len(p_ids)


_G = {}


def _tok_one(arg):
    i, r = arg
    tok, pool, fs_prob, seed, max_len = _G["tok"], _G["pool"], _G["p"], _G["seed"], _G["max_len"]
    rng = random.Random(seed * 1000003 + i)
    fs = None
    if pool and rng.random() < fs_prob:
        fs = rng.sample(pool, rng.randint(1, 4))
    ids, np_ = build_text(tok, r["question"], r["solution"], fs)
    if len(ids) > max_len:
        return None
    return ids, np_


def _init_pool(tok, pool, p, seed, max_len):
    _G.update(tok=tok, pool=pool, p=p, seed=seed, max_len=max_len)


class SFTData(Dataset):
    """Pre-tokenized SFT dataset with prompt tokens masked out of the loss."""

    def __init__(self, rows, tok, max_len, fewshot_pool, fewshot_prob, seed=0, nproc=16):
        from multiprocessing import Pool
        _init_pool(tok, fewshot_pool, fewshot_prob, seed, max_len)
        with Pool(nproc, initializer=_init_pool,
                  initargs=(tok, fewshot_pool, fewshot_prob, seed, max_len)) as p:
            res = p.map(_tok_one, list(enumerate(rows)), chunksize=256)
        self.examples = [r for r in res if r is not None]
        self.lengths = [len(e[0]) for e in self.examples]
        print(f"tokenized {len(self.examples)}/{len(rows)} (dropped {len(rows)-len(self.examples)} "
              f"over max_len); mean_len={sum(self.lengths)/len(self.lengths):.0f}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, np_ = self.examples[i]
        labels = list(ids)
        for j in range(min(np_, len(labels))):
            labels[j] = -100
        return {"input_ids": list(ids), "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        mx = max(len(f["input_ids"]) for f in feats)
        mx = int(math.ceil(mx / 16) * 16)
        input_ids, labels, attn = [], [], []
        for f in feats:
            n = len(f["input_ids"])
            pad = mx - n
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}


def gsm_fewshot_pool(n=200, seed=42):
    """Few-shot demos in the exact style the eval harness uses (from GSM8K TRAIN)."""
    from datasets import load_dataset
    import re
    ds = load_dataset("openai/gsm8k", "main")["train"]
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), n)
    out = []
    for i in idx:
        rec = ds[i]
        body, _, final = rec["answer"].rpartition("####")
        out.append(f"{rec['question']}\n\nReasoning:\n{body.strip()}\n\nANSWER: {final.strip()}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="work/sft1.jsonl")
    ap.add_argument("--out", default="work/sft1_model")
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--fewshot-prob", type=float, default=0.12)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    with open("templates/gemma3.jinja") as f:
        tok.chat_template = f.read()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    print("rows", len(rows))

    pool = gsm_fewshot_pool() if args.fewshot_prob > 0 else None
    ds = SFTData(rows, tok, args.max_len, pool, args.fewshot_prob, seed=args.seed)

    cfg = AutoConfig.from_pretrained(args.init)
    tcfg = getattr(cfg, "text_config", cfg)
    tcfg.torch_dtype = torch.bfloat16
    model = Gemma3ForCausalLM.from_pretrained(args.init, config=tcfg,
                                              dtype=torch.bfloat16,
                                              attn_implementation=os.environ.get("ATTN","eager"))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        optim="adamw_bnb_8bit",
        report_to=[],
        dataloader_num_workers=8,
        seed=args.seed,
        gradient_checkpointing=False,
        use_liger_kernel=os.environ.get("LIGER", "1") == "1",
        average_tokens_across_devices=False,
        group_by_length=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(tok.pad_token_id))
    # feed precomputed lengths to the length-grouped sampler (avoids re-iterating the dataset)
    from transformers.trainer_pt_utils import LengthGroupedSampler
    mega = targs.train_batch_size * targs.gradient_accumulation_steps
    trainer._get_train_sampler = lambda *a, **k: LengthGroupedSampler(
        mega, lengths=ds.lengths, generator=torch.Generator().manual_seed(args.seed))
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    with open("templates/gemma3.jinja") as f:
        ct = f.read()
    tc = json.load(open(os.path.join(args.out, "tokenizer_config.json")))
    tc["chat_template"] = ct
    json.dump(tc, open(os.path.join(args.out, "tokenizer_config.json"), "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
