#!/usr/bin/env python3
"""Extract the text-only decoder (Gemma3ForCausalLM) from the gemma-3-4b-pt snapshot."""
import os, torch
from transformers import (AutoTokenizer, Gemma3ForCausalLM,
                          Gemma3ForConditionalGeneration, GenerationConfig)

SRC = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
DST = "base_text"

full = Gemma3ForConditionalGeneration.from_pretrained(SRC, dtype=torch.bfloat16)
sd = full.state_dict()
print("top-level keys sample:", list(sd)[:3], "...", len(sd))

txt_sd = {}
for k, v in sd.items():
    if k.startswith("model.language_model."):
        txt_sd["model." + k[len("model.language_model."):]] = v
    elif k == "lm_head.weight":
        txt_sd[k] = v
print("text keys:", len(txt_sd))

cfg = full.config.text_config
cfg.torch_dtype = "bfloat16"
model = Gemma3ForCausalLM(cfg)
missing, unexpected = model.load_state_dict(txt_sd, strict=False)
print("missing:", missing)
print("unexpected:", unexpected)
assert not unexpected
assert all("rotary" in m or "lm_head" in m for m in missing), missing
model = model.to(torch.bfloat16)

# sanity: tied lm_head
print("tied:", torch.equal(model.lm_head.weight, model.model.embed_tokens.weight))

model.generation_config = GenerationConfig(
    bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
    cache_implementation="hybrid", do_sample=True, top_k=64, top_p=0.95,
)
os.makedirs(DST, exist_ok=True)
model.save_pretrained(DST, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(SRC)
tok.save_pretrained(DST)
print("saved to", DST)
