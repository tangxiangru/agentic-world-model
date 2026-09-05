#!/usr/bin/env python3
"""LoRA SFT of gemma-3-4b-pt on GSM8K train, rendered with templates/gemma3.jinja."""
import json, math, os, random, sys, time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
OUT = sys.argv[1] if len(sys.argv) > 1 else "out_sft"
MAX_SEQ = 2048
LR = 1e-4
EPOCHS = float(os.environ.get("EPOCHS","1.0"))
TOK_PER_BATCH = 3072
GRAD_ACCUM = 3
SEED = 0
MAX_MIN = float(os.environ.get("MAX_MIN", "13"))

random.seed(SEED); torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained(SNAP)
EOT = "<end_of_turn>"

rows = [json.loads(l) for l in open("train_data.jsonl")]

def render(r):
    # byte-for-byte the gemma3.jinja rendering used by the grader
    user = r["prompt"].strip()
    if r.get("system"):
        user = r["system"].strip() + "\n\n" + user
    prefix = f"{tok.bos_token}<start_of_turn>user\n{user}{EOT}\n<start_of_turn>model\n"
    assert r["completion"].endswith(EOT)
    target = f"{r['completion']}\n"
    return prefix, target

examples, n_trunc = [], 0
for r in rows:
    p, t = render(r)
    pi = tok(p, add_special_tokens=False)["input_ids"]
    ti = tok(t, add_special_tokens=False)["input_ids"]
    if len(pi) + len(ti) > MAX_SEQ:
        n_trunc += 1
        continue
    examples.append((pi + ti, len(pi)))
lens = sorted(len(e[0]) for e in examples)
print(f"n={len(examples)} dropped_too_long={n_trunc} p50={lens[len(lens)//2]} max={lens[-1]}", flush=True)

# length-bucketed batches
examples.sort(key=lambda e: len(e[0]))
batches, cur = [], []
for ex in examples:
    trial = cur + [ex]
    if len(trial) * len(trial[-1][0]) > TOK_PER_BATCH and cur:
        batches.append(cur); cur = [ex]
    else:
        cur = trial
if cur: batches.append(cur)
random.shuffle(batches)
import math as _m
nsteps = int(len(batches) * EPOCHS)
batches = (batches * _m.ceil(EPOCHS))[:nsteps]
print("batches", len(batches), flush=True)

model = Gemma3ForConditionalGeneration.from_pretrained(SNAP, dtype=torch.bfloat16, attn_implementation="eager")
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lcfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                  target_modules=targets, exclude_modules=r".*vision_tower.*|.*multi_modal_projector.*")
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()
model.cuda()

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
nopt = max(1, len(batches) // GRAD_ACCUM)
sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * nopt)), nopt)

PAD = tok.pad_token_id
t0 = time.time()
model.train()
for step, batch in enumerate(batches):
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), PAD, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (seq, plen) in enumerate(batch):
        ids[i, :len(seq)] = torch.tensor(seq)
        att[i, :len(seq)] = 1
        lab[i, plen:len(seq)] = torch.tensor(seq[plen:])
    ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
    out = model(input_ids=ids, attention_mask=att, labels=lab)
    (out.loss / GRAD_ACCUM).backward()
    if (step + 1) % GRAD_ACCUM == 0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if (time.time() - t0) / 60 > MAX_MIN:
        print(f"wall-clock cap {MAX_MIN}m hit at step {step}/{len(batches)}", flush=True)
        break
    if step % 20 == 0:
        el = time.time() - t0
        print(f"step {step}/{len(batches)} loss {out.loss.item():.4f} elapsed {el/60:.1f}m eta {el/(step+1)*(len(batches)-step-1)/60:.1f}m", flush=True)

print("training done", (time.time()-t0)/60, flush=True)
model = model.merge_and_unload()
model.config.use_cache = True
os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
try:
    AutoProcessor.from_pretrained(SNAP).save_pretrained(OUT)
except Exception as e:
    print("processor save skipped:", e)
print("saved", OUT, flush=True)
