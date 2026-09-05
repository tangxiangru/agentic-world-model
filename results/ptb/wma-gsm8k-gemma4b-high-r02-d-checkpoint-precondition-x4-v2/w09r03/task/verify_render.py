"""Render one eval item both ways and diff them.

Way A: exactly what evaluate.py causes -- inspect_evals' solver chain, then
       templates/gemma3.jinja applied by the tokenizer (what vLLM receives).
Way B: gsm_format.render_prompt, which train_data.py uses to build targets.

They must be byte-identical. Also prints token-length stats for the 10-shot
prompt so max_seq_len can be set with real numbers.
"""

from __future__ import annotations

import sys

from transformers import AutoTokenizer

import gsm_format as G

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def main() -> None:
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

    fewshots = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=42,
        limit=10,
    )
    system = "\n\n".join(sample_to_fewshot(s) for s in fewshots)

    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": G.MATH_PROMPT_TEMPLATE.format(prompt=question)},
    ]

    tok = AutoTokenizer.from_pretrained(SNAP)
    with open("templates/gemma3.jinja") as f:
        tmpl = f.read()
    way_a = tok.apply_chat_template(
        messages, chat_template=tmpl, tokenize=False, add_generation_prompt=True
    )

    shots = [(s.input, s.metadata["reasoning"], s.target) for s in fewshots]
    way_b = G.render_prompt(question, fewshots=shots)

    print("template sha256:", G.template_sha256())
    print("identical:", way_a == way_b)
    if way_a != way_b:
        for i, (x, y) in enumerate(zip(way_a, way_b)):
            if x != y:
                print("first diff at", i)
                print("A:", repr(way_a[max(0, i - 120) : i + 120]))
                print("B:", repr(way_b[max(0, i - 120) : i + 120]))
                break
        print("len A", len(way_a), "len B", len(way_b))
        sys.exit(1)

    ids = tok(way_a, add_special_tokens=False)["input_ids"]
    print("10-shot prompt tokens:", len(ids))
    print("first 12 ids:", ids[:12])
    print("decoded head:", repr(tok.decode(ids[:12])))
    zero = G.render_prompt(question)
    print("0-shot prompt tokens:", len(tok(zero, add_special_tokens=False)["input_ids"]))
    print("eot id:", tok.convert_tokens_to_ids("<end_of_turn>"), "eos id:", tok.eos_token_id,
          "bos id:", tok.bos_token_id)
    # the string must not double the bos when add_special_tokens=True is used anywhere
    print("bos in string:", way_a.startswith("<bos>"))


if __name__ == "__main__":
    main()
