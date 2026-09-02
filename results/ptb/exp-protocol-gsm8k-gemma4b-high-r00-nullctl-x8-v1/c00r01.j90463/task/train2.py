import json, os, random, torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

random.seed(0)
SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
MAXLEN = 1024

tok = AutoTokenizer.from_pretrained(SNAP)
tok.chat_template = open("templates/gemma3.jinja").read()

rft = [json.loads(l) for l in open("rft_data.jsonl")]
orig = [json.loads(l) for l in open("train_data.jsonl")]
solved = {r["user"] for r in rft}
covered = {r["user"] for r in rft}
# add human solutions for questions the model could not solve (from the sampled pool),
# plus a slice of the remaining original data for coverage
NQ = int(os.environ.get("NQ", 2600))
pool = orig[:NQ]
extra = [r for r in pool if r["user"] not in solved]
rest = orig[NQ:NQ + 1500]
rows = rft + extra + rest
random.shuffle(rows)
print("rft", len(rft), "unsolved-human", len(extra), "rest", len(rest), "total", len(rows))


def build(r):
    msgs = []
    if r.get("system"):
        msgs.append({"role": "system", "content": r["system"]})
    msgs.append({"role": "user", "content": r["user"]})
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = prompt + r["assistant"].strip() + "<end_of_turn>\n"
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    f = tok(full, add_special_tokens=False)["input_ids"]
    return f[:MAXLEN], ([-100] * len(p) + f[len(p):])[:MAXLEN]


class DS(Dataset):
    def __init__(self, rows):
        self.ex = [build(r) for r in rows]
        self.ex = [e for e in self.ex if any(l != -100 for l in e[1])]

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return {"input_ids": self.ex[i][0], "labels": self.ex[i][1]}


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
print("examples", len(ds))
model = AutoModelForCausalLM.from_pretrained("model_v1", dtype=torch.bfloat16,
                                             attn_implementation="eager")
model.config.use_cache = False
for n, p in model.named_parameters():
    if "vision_tower" in n or "multi_modal_projector" in n:
        p.requires_grad = False

args = TrainingArguments(
    output_dir="out2", per_device_train_batch_size=8, gradient_accumulation_steps=2,
    num_train_epochs=1.0, learning_rate=7e-6, lr_scheduler_type="cosine",
    warmup_ratio=0.03, logging_steps=20, bf16=True, optim="adamw_bnb_8bit",
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    save_strategy="no", report_to=[], group_by_length=True, length_column_name=None,
    max_grad_norm=1.0, seed=0)

trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
trainer.train()
model.config.use_cache = True
trainer.save_model("model_v2")
tok.save_pretrained("model_v2")
import shutil
for f in ["preprocessor_config.json", "processor_config.json"]:
    s = os.path.join(SNAP, f)
    if os.path.exists(s):
        shutil.copy(s, os.path.join("model_v2", f))
print("saved model_v2")
