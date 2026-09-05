#!/usr/bin/env python3
"""Full SFT of google/gemma-3-4b-pt (text-only Gemma3ForCausalLM) on GSM8K-style
completion-only data. Targets end with <end_of_turn> (id 106) so vLLM stops there.
Renders prompts with the grader's templates/gemma3.jinja (byte-for-byte)."""
import argparse, json, os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, Gemma3ForCausalLM, Trainer,
                          TrainingArguments, set_seed)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
END_OF_TURN = 106

class SFTData(Dataset):
    def __init__(self, path, tok, max_len):
        self.ex = []
        rows = [json.loads(l) for l in open(path)]
        n_trunc = 0
        for d in rows:
            ptext = tok.apply_chat_template([{"role": "user", "content": d["prompt"]}],
                                            tokenize=False, add_generation_prompt=True)
            pids = tok(ptext, add_special_tokens=False)["input_ids"]
            cids = tok(d["completion"], add_special_tokens=False)["input_ids"]
            assert cids[-1] == END_OF_TURN, cids[-3:]
            ids = pids + cids
            labels = [-100] * len(pids) + list(cids)
            if len(ids) > max_len:
                n_trunc += 1
                ids = ids[:max_len]; labels = labels[:max_len]
            self.ex.append({"input_ids": ids, "labels": labels})
        print(f"loaded {len(self.ex)} rows, truncated {n_trunc} ({100*n_trunc/len(self.ex):.2f}%)")
    def __len__(self): return len(self.ex)
    def __getitem__(self, i): return self.ex[i]

class Collator:
    def __init__(self, pad_id): self.pad_id = pad_id
    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            p = m - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * p)
            labels.append(b["labels"] + [-100] * p)
            attn.append([1] * len(b["input_ids"]) + [0] * p)
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/gsm8k_train.jsonl")
    ap.add_argument("--out", default="ckpts/exp01")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(SNAP)
    tok.chat_template = open("templates/gemma3.jinja").read()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    ds = SFTData(args.data, tok, args.max_len)
    # sanity: every example's last label token is the stop token
    assert all(e["labels"][-1] == END_OF_TURN for e in ds.ex if len(e["labels"]) < args.max_len)

    if args.dry_run:
        print("DRY RUN ok; example decoded completion tail:")
        e = ds.ex[0]
        tail = [t for t in e["labels"] if t != -100]
        print(repr(tok.decode(tail)))
        return

    model = Gemma3ForCausalLM.from_pretrained(SNAP, torch_dtype=torch.bfloat16,
                                              attn_implementation="eager")
    model.config.use_cache = False
    if model.generation_config is not None:
        model.generation_config.eos_token_id = [1, 106]
        # greedy decoding is better for math (see exp-01: +9.3 pts); vLLM uses this at eval
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        weight_decay=0.0, bf16=True, logging_steps=20, save_strategy="epoch",
        save_total_limit=1, report_to=[], gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, seed=args.seed,
        dataloader_num_workers=4, optim="adamw_torch",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(pad_id))
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    model.generation_config.save_pretrained(final)
    print("saved to", final)

if __name__ == "__main__":
    main()
