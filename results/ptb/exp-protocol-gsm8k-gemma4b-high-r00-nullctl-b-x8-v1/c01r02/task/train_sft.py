#!/usr/bin/env python3
"""Full fine-tune of gemma-3-4b-pt (text tower) on math CoT with completion-only loss."""
from __future__ import annotations
import argparse, json, math, os, random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, Gemma3ForCausalLM, Trainer, TrainingArguments,
                          set_seed)

BOS = '<bos>'
SOT = '<start_of_turn>'
EOT = '<end_of_turn>'


def build_texts(rec):
    """Returns (prompt_text, completion_text) following templates/gemma3.jinja."""
    user = rec['user'].strip()
    if rec.get('system'):
        user = rec['system'].strip() + '\n\n' + user
    prompt = f"{BOS}{SOT}user\n{user}{EOT}\n{SOT}model\n"
    completion = rec['assistant'].strip() + f"{EOT}\n"
    return prompt, completion


class SFTData(Dataset):
    def __init__(self, path, tok, max_len, limit=None):
        self.tok = tok
        self.max_len = max_len
        self.ex = []
        skipped = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                r = json.loads(line)
                p, c = build_texts(r)
                pi = tok(p, add_special_tokens=False)['input_ids']
                ci = tok(c, add_special_tokens=False)['input_ids']
                if len(pi) + len(ci) > max_len:
                    skipped += 1
                    continue
                self.ex.append((pi, ci))
        self.lengths = [len(a) + len(b) for a, b in self.ex]
        print(f'dataset: {len(self.ex)} examples, skipped {skipped}, '
              f'tokens {sum(self.lengths)/1e6:.1f}M, mean {np.mean(self.lengths):.0f}')

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        p, c = self.ex[i]
        ids = p + c
        labels = [-100] * len(p) + list(c)
        return {'input_ids': ids, 'labels': labels, 'length': len(ids)}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f['input_ids']) for f in feats)
        n = ((n + 7) // 8) * 8
        input_ids, labels, attn = [], [], []
        for f in feats:
            k = n - len(f['input_ids'])
            input_ids.append(f['input_ids'] + [self.pad_id] * k)
            labels.append(f['labels'] + [-100] * k)
            attn.append([1] * len(f['input_ids']) + [0] * k)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attn, dtype=torch.long),
        }


def _ce_chunk(hs, ls, lm_head):
    logits = lm_head(hs).float()
    return torch.nn.functional.cross_entropy(logits, ls, reduction='sum')


class ChunkedCETrainer(Trainer):
    """Memory-efficient loss: only completion tokens, logits recomputed in chunks."""
    chunk = 4096

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels')
        base = model.module if hasattr(model, 'module') else model
        hidden = base.model(input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask']).last_hidden_state
        h = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
        lab = labels[:, 1:].reshape(-1)
        keep = lab != -100
        h = h[keep]
        lab = lab[keep]
        n = max(lab.numel(), 1)
        total = None
        for i in range(0, h.size(0), self.chunk):
            li = torch.utils.checkpoint.checkpoint(
                _ce_chunk, h[i:i + self.chunk], lab[i:i + self.chunk],
                base.lm_head, use_reentrant=False)
            total = li if total is None else total + li
        if total is None:
            total = hidden.sum() * 0.0
        return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='base_text')
    ap.add_argument('--data', default='data/sft.jsonl')
    ap.add_argument('--out', default='runs/sft1')
    ap.add_argument('--max-len', type=int, default=2048)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--epochs', type=float, default=2.0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--max-steps', type=int, default=-1)
    ap.add_argument('--warmup', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-steps', type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    ds = SFTData(args.data, tok, args.max_len, args.limit)

    model = Gemma3ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type='cosine',
        warmup_steps=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        logging_steps=10,
        bf16=True,
        optim='adamw_torch_fused',
        group_by_length=True,
        length_column_name='length',
        save_strategy='steps' if args.save_steps else 'no',
        save_steps=args.save_steps or 10 ** 9,
        save_total_limit=2,
        report_to=[],
        dataloader_num_workers=4,
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = ChunkedCETrainer(model=model, args=targs, train_dataset=ds,
                               data_collator=Collator(tok.pad_token_id))
    trainer.model_accepts_loss_kwargs = False
    trainer.train()
    model.config.use_cache = True
    trainer.save_model(args.out + '/final')
    tok.save_pretrained(args.out + '/final')
    # keep the pt generation config (stops on <end_of_turn>)
    import shutil
    shutil.copy(os.path.join(args.model, 'generation_config.json'),
                os.path.join(args.out, 'final', 'generation_config.json'))
    print('saved', args.out + '/final')


if __name__ == '__main__':
    main()
