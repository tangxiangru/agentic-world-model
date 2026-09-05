import json, os, sys, argparse, random
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Gemma3ForConditionalGeneration,
                          Trainer, TrainingArguments)

SNAP='/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d'

class JsonlDS(Dataset):
    def __init__(self, path, max_len):
        self.data=[]
        for line in open(path):
            line=line.strip()
            if not line: continue
            r=json.loads(line)
            ids=r['input_ids'][:max_len]; labs=r['labels'][:max_len]
            self.data.append((ids,labs))
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        ids,labs=self.data[i]
        return {"input_ids":ids,"labels":labs}

class Collator:
    def __init__(self,pad_id): self.pad_id=pad_id
    def __call__(self,batch):
        maxlen=max(len(b['input_ids']) for b in batch)
        input_ids=[]; labels=[]; attn=[]
        for b in batch:
            ids=b['input_ids']; labs=b['labels']
            pad=maxlen-len(ids)
            input_ids.append(ids+[self.pad_id]*pad)
            labels.append(labs+[-100]*pad)
            attn.append([1]*len(ids)+[0]*pad)
        return {"input_ids":torch.tensor(input_ids),
                "labels":torch.tensor(labels),
                "attention_mask":torch.tensor(attn)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='train_tok.jsonl')
    ap.add_argument('--out',default='ft_out')
    ap.add_argument('--epochs',type=float,default=2.0)
    ap.add_argument('--lr',type=float,default=1e-5)
    ap.add_argument('--bs',type=int,default=8)
    ap.add_argument('--accum',type=int,default=4)
    ap.add_argument('--max_len',type=int,default=1536)
    ap.add_argument('--warmup',type=float,default=0.03)
    args=ap.parse_args()

    tok=AutoTokenizer.from_pretrained(SNAP)
    ds=JsonlDS(args.data,args.max_len)
    print('dataset size',len(ds))

    model=Gemma3ForConditionalGeneration.from_pretrained(
        SNAP, torch_dtype=torch.bfloat16, attn_implementation='eager')
    # freeze vision components; train language model only
    frozen=0; trainable=0
    for n,p in model.named_parameters():
        if n.startswith('model.vision_tower') or n.startswith('model.multi_modal_projector') \
           or n.startswith('vision_tower') or n.startswith('multi_modal_projector'):
            p.requires_grad_(False); frozen+=p.numel()
        else:
            trainable+=p.numel()
    print(f'trainable {trainable/1e9:.2f}B frozen {frozen/1e9:.2f}B')
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()

    targs=TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        warmup_ratio=args.warmup,
        logging_steps=20,
        save_strategy='no',
        bf16=True,
        optim='adamw_bnb_8bit',
        weight_decay=0.0,
        max_grad_norm=1.0,
        report_to=[],
        dataloader_num_workers=4,
        gradient_checkpointing=False,  # already enabled manually
    )
    trainer=Trainer(model=model,args=targs,train_dataset=ds,
                    data_collator=Collator(tok.pad_token_id))
    trainer.train()
    print('saving to',args.out)
    model.config.use_cache=True
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print('DONE')

if __name__=='__main__':
    main()
