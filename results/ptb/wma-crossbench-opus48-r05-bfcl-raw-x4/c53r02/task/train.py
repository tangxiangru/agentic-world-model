import json, os, sys, shutil, argparse
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Gemma3ForConditionalGeneration,
                          Trainer, TrainingArguments, DataCollatorForSeq2Seq)

SNAP='/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', default='xlam_single.jsonl')
    ap.add_argument('--out', default='out_sft')
    ap.add_argument('--epochs', type=float, default=2.0)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--maxlen', type=int, default=2048)
    ap.add_argument('--limit', type=int, default=0)
    args=ap.parse_args()

    tok=AutoTokenizer.from_pretrained(SNAP)
    tok.chat_template=open('templates/gemma3_tool_calling.jinja').read()

    rows=[json.loads(l) for l in open(args.data)]
    if args.limit:
        rows=rows[:args.limit]
    print('rows:', len(rows))

    def build(r):
        user={'role':'user','content':r['query']}
        asst={'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':r['name'],'arguments':json.dumps(r['arguments'])}}]}
        full=tok.apply_chat_template([user,asst],tools=r['tools'],add_generation_prompt=False,tokenize=False)
        prompt=tok.apply_chat_template([user],tools=r['tools'],add_generation_prompt=True,tokenize=False)
        return full, prompt

    def tokenize_row(r):
        full, prompt = build(r)
        if not full.startswith(prompt):
            return None
        full_ids=tok(full, add_special_tokens=False)['input_ids']
        prompt_ids=tok(prompt, add_special_tokens=False)['input_ids']
        if full_ids[:len(prompt_ids)] != prompt_ids:
            # fall back: skip mismatched (rare)
            return None
        if len(full_ids) > args.maxlen:
            return None
        labels=[-100]*len(prompt_ids)+full_ids[len(prompt_ids):]
        assert len(labels)==len(full_ids)
        return {'input_ids':full_ids,'labels':labels,'attention_mask':[1]*len(full_ids)}

    feats={'input_ids':[],'labels':[],'attention_mask':[]}
    skipped=0
    for r in rows:
        t=tokenize_row(r)
        if t is None:
            skipped+=1; continue
        for k in feats: feats[k].append(t[k])
    print('tokenized:', len(feats['input_ids']), 'skipped:', skipped)
    lens=[len(x) for x in feats['input_ids']]
    print('len stats: max',max(lens),'mean',sum(lens)//len(lens))
    ds=Dataset.from_dict(feats)

    model=Gemma3ForConditionalGeneration.from_pretrained(
        SNAP, torch_dtype=torch.bfloat16, attn_implementation='eager')
    model.config.use_cache=False
    model.gradient_checkpointing_enable()

    collator=DataCollatorForSeq2Seq(tok, model=None, padding='longest', label_pad_token_id=-100)

    targs=TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=20,
        bf16=True,
        optim='paged_adamw_8bit',
        save_strategy='no',
        report_to=[],
        gradient_checkpointing=True,
        group_by_length=True,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
    )

    trainer=Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    # copy aux files from snapshot (dereference), then save trained weights+config+tokenizer
    for fn in os.listdir(SNAP):
        if fn.endswith('.safetensors') or fn=='model.safetensors.index.json' or fn=='config.json':
            continue
        src=os.path.join(SNAP,fn)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.out,fn))
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print('SAVED to', args.out)

if __name__=='__main__':
    main()
