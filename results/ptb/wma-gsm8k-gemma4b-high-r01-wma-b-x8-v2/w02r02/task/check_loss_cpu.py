#!/usr/bin/env python3
"""CPU smoke test: does the exact label construction used by train_sft.py give a
sensible, non-zero completion-only loss under the base model?

Catches the seq_len_truncation / all-masked-labels class of failure, where the
run exits 0 with a flat or zero loss. Runs on CPU so it never touches the GPU
a training job is using.
"""
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from train_sft import SNAPSHOT, Collator, SFTData, load_template

template, thash = load_template()
tok = AutoTokenizer.from_pretrained(SNAPSHOT)
ds = SFTData("data/sft_train.jsonl", tok, template, 1536, limit=8, verbose=False)
coll = Collator(tok.pad_token_id or 0)
batch = coll([ds[i] for i in range(4)])

n_lab = int((batch["labels"] != -100).sum())
print(f"[labels] {n_lab} supervised tokens of {batch['labels'].numel()} "
      f"({100 * n_lab / batch['labels'].numel():.1f}%)")
assert n_lab > 0, "no supervised tokens -> the run would train on nothing"

model = Gemma3ForConditionalGeneration.from_pretrained(
    SNAPSHOT, dtype=torch.float32, attn_implementation="eager"
)
model.eval()
with torch.no_grad():
    out = model(**batch)
print(f"[loss] base-model completion-only loss = {out.loss.item():.4f} nats")
# a pretrained LM on in-domain english math prose sits well inside this band
assert 0.2 < out.loss.item() < 6.0, "loss outside the plausible band"
print("[ok] label masking and loss path are sane")
