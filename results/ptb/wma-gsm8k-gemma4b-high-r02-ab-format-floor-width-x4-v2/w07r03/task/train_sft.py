#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on a prompt/completion jsonl.

The jsonl rows are ALREADY rendered with templates/gemma3.jinja (verified
byte-for-byte against tokenizer.apply_chat_template in build_data.py), so this
script only tokenises, masks the prompt, and trains.

  ids    = [bos] + tok(prompt) + tok(completion)
  labels = [-100]*(1+len(prompt))  + tok(completion)

The completion always ends in <end_of_turn> (id 106), which is in the
snapshot's generation_config.eos_token_id, so vLLM stops there at grading.
"""
import argparse, json, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoProcessor, AutoConfig,
                          Gemma3ForConditionalGeneration, Trainer,
                          TrainingArguments, set_seed)

SNAP = os.environ.get("PTB_BASE_MODEL_SNAPSHOT")


class SFTData(Dataset):
    def __init__(self, path, tok, max_len, limit=None, seed=0):
        rows = [json.loads(l) for l in open(path)]
        if limit is not None and limit < len(rows):
            rng = np.random.default_rng(seed)
            rows = [rows[i] for i in rng.permutation(len(rows))[:limit]]
        P = tok([r["prompt"] for r in rows], add_special_tokens=False)["input_ids"]
        C = tok([r["completion"] for r in rows], add_special_tokens=False)["input_ids"]
        bos = tok.bos_token_id
        self.ex, n_trunc = [], 0
        for p, c in zip(P, C):
            ids = [bos] + p + c
            n_lab = len(c)
            if len(ids) > max_len:          # never truncate the completion
                n_trunc += 1
                continue
            self.ex.append((ids, n_lab))
        self.lengths = [len(i) for i, _ in self.ex]
        print(f"[data] {len(self.ex)} rows kept, {n_trunc} dropped (> {max_len} tok), "
              f"mean {np.mean(self.lengths):.0f} tok, max {max(self.lengths)}", flush=True)

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        ids, n_lab = self.ex[i]
        labels = [-100] * (len(ids) - n_lab) + ids[len(ids) - n_lab:]
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class Collator:
    def __init__(self, pad_id, multiple_of=16):
        self.pad_id, self.m = pad_id, multiple_of

    def __call__(self, feats):
        L = max(len(f["input_ids"]) for f in feats)
        L = ((L + self.m - 1) // self.m) * self.m
        ids = torch.full((len(feats), L), self.pad_id, dtype=torch.long)
        lab = torch.full((len(feats), L), -100, dtype=torch.long)
        att = torch.zeros((len(feats), L), dtype=torch.long)
        for i, f in enumerate(feats):
            n = len(f["input_ids"])
            ids[i, :n] = torch.tensor(f["input_ids"])
            lab[i, :n] = torch.tensor(f["labels"])
            att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


class SparseLossTrainer(Trainer):
    """Apply the LM head only where a label is supervised.

    Gemma3's vocab is 262 208, so materialising logits for every position of a
    16 x 2560 batch costs ~21 GB in bf16 alone and OOMs an 80 GB H100. Under
    completion-only loss ~2/3 of positions carry -100, and dropping them makes
    the head's matmul (2560 x 262208) proportionally cheaper too.
    config.final_logit_softcapping is None on this checkpoint, so the head is
    exactly nn.Linear with no post-scaling to reproduce.
    """

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None, **kw):
        labels = inputs.pop("labels")
        base = model.module if hasattr(model, "module") else model
        hidden = base.model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"]).last_hidden_state
        tgt = labels[:, 1:]
        sel = tgt != -100
        h = hidden[:, :-1, :][sel]                       # (n_supervised, hidden)
        logits = base.lm_head(h).float()
        loss = torch.nn.functional.cross_entropy(logits, tgt[sel], reduction="sum")
        # main() forces model_accepts_loss_kwargs=False, so Trainer passes
        # num_items_in_batch=None and does the /grad_accum division itself.
        # The other branch is kept correct in case that ever changes.
        loss = loss / (num_items_in_batch if num_items_in_batch is not None
                       else sel.sum())
        return (loss, None) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_pool.jsonl")
    ap.add_argument("--parent", default=SNAP)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--save-total-limit", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.parent)
    ds = SFTData(args.data, tok, args.max_seq_len, args.limit, args.seed)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation=args.attn)
    # text-only task: the SigLIP tower and the projector never see a gradient
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    # A parent produced by THIS script carries the greedy generation_config
    # (do_sample False + temperature 0.0) that vLLM needs. transformers refuses
    # to save that combination -- GenerationConfig.save_pretrained() raises
    # "`do_sample` is set to `False`. However, `temperature` is set to `0.0`" --
    # and Trainer._save goes through it at every checkpoint, so a continuation
    # run would die at the first save with no weights written. Reset it to a
    # valid default here; the json.dump at the end rewrites the greedy file.
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {n_train/1e9:.2f}B, frozen {n_frozen/1e6:.0f}M", flush=True)
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        optim=args.optim,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=args.save_total_limit,
        group_by_length=True,
        length_column_name="length",
        gradient_checkpointing=args.grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    trainer = SparseLossTrainer(model=model, args=targs, train_dataset=ds,
                                data_collator=Collator(tok.pad_token_id))
    # deterministic gradient-accumulation normalisation: with this False the
    # Trainer passes num_items_in_batch=None and divides the loss by
    # gradient_accumulation_steps itself, instead of skipping that division
    # (transformers >=4.46 behaviour) and making every update 8x too large.
    trainer.model_accepts_loss_kwargs = False
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    try:
        AutoProcessor.from_pretrained(args.parent).save_pretrained(final)
    except Exception as e:
        print("[warn] processor save failed:", e)
    # Greedy decoding. evaluate.py passes no temperature, and vLLM's
    # get_diff_sampling_param() (vllm/config/model.py:1367) reads exactly
    # {repetition_penalty, temperature, top_k, top_p, min_p, max_new_tokens}
    # from generation_config.json - do_sample is NOT among them. So temperature
    # must be 0.0, and top_k/top_p are omitted so they fall back to vLLM's
    # neutral defaults instead of the shipped 64 / 0.95.
    gc = {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
          "cache_implementation": "hybrid", "do_sample": False,
          "temperature": 0.0, "transformers_version": "4.57.3"}
    json.dump(gc, open(os.path.join(final, "generation_config.json"), "w"), indent=2)
    print("[done] saved", final, flush=True)


if __name__ == "__main__":
    main()
