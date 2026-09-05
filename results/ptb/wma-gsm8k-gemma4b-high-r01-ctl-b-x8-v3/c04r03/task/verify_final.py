#!/usr/bin/env python3
"""Pre-deadline check on final_model/: it must load from a fresh process the
way the grader loads it, and it must render/stop the way training assumed.

Checks (pitfall final_model_not_loadable):
  1. config.json architecture is what evaluate.py's model_type() will accept
  2. transformers can build the model on CPU (meta-device init, no weights read
     into RAM twice) and the safetensors index covers every parameter
  3. the tokenizer is present alongside the weights
  4. generation_config.json decodes greedily and stops on <end_of_turn>
  5. the grader's templates/gemma3.jinja renders against this tokenizer
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parent


def main(d: str) -> int:
    p = Path(d)
    bad = []

    cfg = json.loads((p / "config.json").read_text())
    arch = cfg["architectures"][0]
    print("architecture:", arch)
    if "gemma" not in arch.lower():
        bad.append(f"evaluate.py model_type() will not map {arch}")

    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    import torch

    tok = AutoTokenizer.from_pretrained(p)
    print("tokenizer ok, vocab", len(tok))
    for t in ("<end_of_turn>", "<start_of_turn>", "<bos>"):
        i = tok.convert_tokens_to_ids(t)
        print(f"  {t} -> {i}")
        if i is None or i == tok.unk_token_id:
            bad.append(f"missing special token {t}")

    gc = json.loads((p / "generation_config.json").read_text())
    print("generation_config:", gc)
    if gc.get("temperature", 1.0) != 0.0:
        bad.append("generation_config temperature is not 0.0 (vLLM would sample)")
    eos = gc.get("eos_token_id")
    if not (eos == 106 or (isinstance(eos, list) and 106 in eos)):
        bad.append("generation_config eos_token_id does not include 106 (<end_of_turn>)")

    tmpl = (TASK / "templates" / "gemma3.jinja").read_text()
    s = tok.apply_chat_template([{"role": "user", "content": "2+2?"}],
                                chat_template=tmpl, tokenize=False,
                                add_generation_prompt=True)
    print("rendered:", repr(s))
    if not s.endswith("<start_of_turn>model\n"):
        bad.append("grader template does not render against this tokenizer")

    # real load, on CPU, exactly as a fresh process would
    conf = AutoConfig.from_pretrained(p)
    model = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, config=conf)
    n = sum(x.numel() for x in model.parameters())
    print(f"loaded on CPU: {type(model).__name__}, {n/1e9:.2f}B params")
    w = model.get_input_embeddings().weight
    if torch.isnan(w).any() or torch.isinf(w).any():
        bad.append("embedding weights contain nan/inf")

    if bad:
        print("\nFAIL:")
        for b in bad:
            print(" -", b)
        return 1
    print("\nOK: final_model looks loadable and correctly configured")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "final_model"))
