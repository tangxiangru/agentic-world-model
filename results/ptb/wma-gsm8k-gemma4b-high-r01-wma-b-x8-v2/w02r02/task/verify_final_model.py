#!/usr/bin/env python3
"""Pre-deadline check on final_model/: does it load from a fresh CPU process the
way the grader's fresh vLLM process will, and does evaluate.py resolve it?

Covers the final_model_not_loadable pitfall: no adapter, tokenizer present,
config.json readable, architecture routes to templates/gemma3.jinja, and the
decode config is the one that was actually measured.
"""
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

FM = Path("final_model")

cfg = json.loads((FM / "config.json").read_text())
arch = cfg["architectures"][0]
print(f"[config] architectures[0]={arch}")
assert "gemma" in arch.lower(), "evaluate.py's model_type() would raise"

gen = json.loads((FM / "generation_config.json").read_text())
print(f"[decode] {gen}")
assert gen.get("temperature") == 0.0 and gen.get("top_p") == 1.0, "not the greedy config that was measured"
assert "top_k" not in gen, "top_k must be absent so vLLM does not override greedy"
assert 106 in gen["eos_token_id"], "<end_of_turn>=106 must be an eos id"

for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "preprocessor_config.json", "processor_config.json",
          "model.safetensors.index.json"]:
    assert (FM / f).exists(), f"missing {f}"
assert not list(FM.glob("adapter_*")), "final_model must not be a LoRA adapter dir"
print("[files] ok:", sorted(p.name for p in FM.iterdir()))

tok = AutoTokenizer.from_pretrained(FM)
assert tok.convert_tokens_to_ids("<end_of_turn>") == 106
print("[tokenizer] ok, <end_of_turn>=106")

model = Gemma3ForConditionalGeneration.from_pretrained(FM, dtype=torch.bfloat16)
n = sum(p.numel() for p in model.parameters())
print(f"[weights] loaded {n/1e9:.2f}B params on CPU")

template = Path("templates/gemma3.jinja").read_text()
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "What is 2 + 2?"}],
    chat_template=template, tokenize=False, add_generation_prompt=True,
)
ids = tok(prompt, add_special_tokens=False, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=64, do_sample=False)
print("[generate]", repr(tok.decode(out[0][ids["input_ids"].shape[1]:])))
print("[ok] final_model is loadable and serves the grader's template")
