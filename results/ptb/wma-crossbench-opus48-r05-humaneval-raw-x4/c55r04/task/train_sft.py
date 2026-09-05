import os, json, argparse, random
os.environ.setdefault("HF_HOME","/home/ben/hf_cache")
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence

SNAP="/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="train_final.jsonl")
    ap.add_argument("--out",default="lora_out")
    ap.add_argument("--n",type=int,default=40000)
    ap.add_argument("--epochs",type=float,default=3.0)
    ap.add_argument("--maxlen",type=int,default=1536)
    ap.add_argument("--bs",type=int,default=16)
    ap.add_argument("--ga",type=int,default=2)
    ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--rank",type=int,default=64)
    ap.add_argument("--attn",default="flash_attention_2")
    ap.add_argument("--gc",type=int,default=0)
    args=ap.parse_args()
    print("PARSED ARGS:",vars(args),flush=True)

    tok=AutoTokenizer.from_pretrained(SNAP)
    print("bos",tok.bos_token,tok.bos_token_id,"eos",tok.eos_token,tok.eos_token_id,"pad",tok.pad_token)
    if tok.pad_token is None:
        tok.pad_token=tok.eos_token

    rows=[json.loads(l) for l in open(args.data)]
    random.seed(0); random.shuffle(rows)
    rows=rows[:args.n]

    PROMPT="<bos><start_of_turn>user\n{ins}<end_of_turn>\n<start_of_turn>model\n"
    EOT="<end_of_turn>\n"
    def build(r):
        ptxt=PROMPT.format(ins=r["instruction"])
        ftxt=ptxt+r["response"]+EOT
        pids=tok(ptxt,add_special_tokens=False)["input_ids"]
        fids=tok(ftxt,add_special_tokens=False)["input_ids"]
        if len(fids)>args.maxlen: return None
        labels=list(fids)
        for i in range(min(len(pids),len(fids))): labels[i]=-100
        return {"input_ids":fids,"labels":labels}
    data=[]
    for r in rows:
        e=build(r)
        if e: data.append(e)
    print("usable examples:",len(data))

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(data)
        def __getitem__(self,i): return data[i]
    def collate(batch):
        ids=[torch.tensor(b["input_ids"]) for b in batch]
        lbl=[torch.tensor(b["labels"]) for b in batch]
        ids=pad_sequence(ids,batch_first=True,padding_value=tok.pad_token_id)
        lbl=pad_sequence(lbl,batch_first=True,padding_value=-100)
        attn=(ids!=tok.pad_token_id).long()
        return {"input_ids":ids,"attention_mask":attn,"labels":lbl}

    model=AutoModelForImageTextToText.from_pretrained(
        SNAP,torch_dtype=torch.bfloat16,attn_implementation=args.attn)
    model.config.use_cache=False
    lora=LoraConfig(
        r=args.rank,lora_alpha=args.rank*2,lora_dropout=0.05,bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    )
    model=get_peft_model(model,lora)
    model.print_trainable_parameters()

    ta=TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.ga,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=bool(args.gc),
        gradient_checkpointing_kwargs={"use_reentrant":False},
        report_to=[],
        dataloader_num_workers=4,
        group_by_length=False,
        optim="adamw_torch",
    )
    tr=Trainer(model=model,args=ta,train_dataset=DS(),data_collator=collate)
    tr.train()
    # merge and save full model
    model=model.merge_and_unload()
    os.makedirs(args.out,exist_ok=True)
    model.save_pretrained(args.out,safe_serialization=True)
    tok.save_pretrained(args.out)
    print("saved to",args.out)

if __name__=="__main__":
    main()
