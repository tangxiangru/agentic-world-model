#!/usr/bin/env python3
"""Verify build_data.render_prompt reproduces templates/gemma3.jinja byte-for-byte.

Pitfall `template_unreachable`: training and grading must render the same string.
This renders the same conversation both ways and diffs them.
"""
import hashlib
import json
import sys

from jinja2 import Environment
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from build_data import MATH_PROMPT_TEMPLATE, fewshot_block, render_prompt

TEMPLATE_PATH = "templates/gemma3.jinja"


def jinja_render(messages, bos_token="<bos>"):
    src = open(TEMPLATE_PATH).read()
    env = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False)

    def raise_exception(msg):
        raise TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    tmpl = env.from_string(src)
    return tmpl.render(messages=messages, bos_token=bos_token,
                       add_generation_prompt=True)


def main():
    print("template sha256:",
          hashlib.sha256(open(TEMPLATE_PATH, "rb").read()).hexdigest()[:16])

    q = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    user = MATH_PROMPT_TEMPLATE.format(prompt=q)
    sysmsg = "\n\n".join([
        fewshot_block("A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?",
                      "It takes 2/2=1 bolt of white fiber\nSo the total amount of fabric is 2+1=3 bolts of fabric", "3"),
        fewshot_block("Weng earns $12 an hour for babysitting.  Yesterday, she just did 50 minutes of babysitting.  How much did she earn?",
                      "Weng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $10.", "10"),
    ])

    ok = True
    for name, system in [("zero-shot", None), ("few-shot", sysmsg)]:
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": user}]
        a = jinja_render(messages)
        b = render_prompt(system, user)
        same = a == b
        ok &= same
        print(f"{name}: match={same}")
        if not same:
            print("--- jinja ---"); print(json.dumps(a))
            print("--- mine  ---"); print(json.dumps(b))

    # also verify the tokenizer round-trips the completion terminator
    from transformers import AutoTokenizer
    snap = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
    tok = AutoTokenizer.from_pretrained(snap)
    ids = tok("x<end_of_turn>", add_special_tokens=False)["input_ids"]
    print("'<end_of_turn>' ->", ids[-1], repr(tok.convert_ids_to_tokens(ids[-1])))
    assert ids[-1] == 106, ids
    bos = tok("<bos>hello", add_special_tokens=False)["input_ids"]
    print("'<bos>' ->", bos[0], repr(tok.convert_ids_to_tokens(bos[0])))
    assert bos[0] == 2, bos
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
