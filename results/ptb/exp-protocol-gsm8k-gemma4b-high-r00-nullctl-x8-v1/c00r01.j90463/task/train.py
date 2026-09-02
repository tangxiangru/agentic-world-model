import json, os, math, random
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments, AutoConfig)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
MAXLEN = 1024

tok = AutoTokenizer.from_pretrained(SNAP)
tok.chat_template = open("templates/gemma3.jinja").read()

rows = [json.loads(l) for l in open("train_data.jsonl")]


def build(r):
    msgs = []
    if r["system"]:
        msgs.append({"role": "system", "content": r["system"]})
    msgs.append({"role": "user", "content": r["user"]})
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = prompt + r["assistant"].strip() + "<end_of_turn>\n"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return f_ids[:MAXLEN], labels[:MAXLEN]


class DS(Dataset):
    def __init__(self, rows):
        self.ex = [build(r) for r in rows]
        self.ex = [e for e in self.ex if any(l != -100 for l in e[1])]

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        ids, lab = self.ex[i]
        return {"input_ids": ids, "labels": lab}


def collate(feats):
    m = max(len(f["input_ids"]) for f in feats)
    pad = tok.pad_token_id or 0
    ii, ll, am = [], [], []
    for f in feats:
        d = m - len(f["input_ids"])
        ii.append(f["input_ids"] + [pad] * d)
        ll.append(f["labels"] + [-100] * d)
        am.append([1] * len(f["input_ids"]) + [0] * d)
    return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
            "attention_mask": torch.tensor(am)}


ds = DS(rows)
print("examples:", len(ds), "avg len", sum(len(e[0]) for e in ds.ex) / len(ds))

model = AutoModelForCausalLM.from_pretrained(
    SNAP, dtype=torch.bfloat16, attn_implementation="eager")
model.config.use_cache = False
# freeze vision tower / multimodal projector (text-only training)
for n, p in model.named_parameters():
    if "vision_tower" in n or "multi_modal_projector" in n:
        p.requires_grad = False

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=float(os.environ.get("EPOCHS", 2)),
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    bf16=True,
    optim="adamw_bnb_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    save_strategy="epoch",
    save_total_limit=3,
    report_to=[],
    group_by_length=True,
    length_column_name=None,
    max_grad_norm=1.0,
    seed=0,
)

trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
trainer.train()

out = "final_model"
model.config.use_cache = True
trainer.save_model(out)
tok.save_pretrained(out)
# copy processor/vision config bits needed by vLLM
import shutil
for f in ["preprocessor_config.json", "processor_config.json"]:
    src = os.path.join(SNAP, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(out, f))
print("saved", out)
