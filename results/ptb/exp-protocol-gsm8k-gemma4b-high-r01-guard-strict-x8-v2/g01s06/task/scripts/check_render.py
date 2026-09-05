#!/usr/bin/env python3
"""Pitfall `template_unreachable`: prove scripts/prompting.py renders exactly what
templates/gemma3.jinja renders for the same messages, and that the target ends on
the token the grader stops at."""
import json
import sys

from jinja2 import Environment
from transformers import AutoTokenizer

sys.path.insert(0, "scripts")
from prompting import fewshot_block, render_prompt, render_target, user_content  # noqa: E402

SNAP = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")

tok = AutoTokenizer.from_pretrained(SNAP)
template_src = open("templates/gemma3.jinja").read()

env = Environment()
env.policies["json.dumps_kwargs"] = {"sort_keys": True}
env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
tpl = env.from_string(template_src)

Q = "Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?"
SYS = "\n\n".join([
    fewshot_block("A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?",
                  "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric", "3"),
    fewshot_block("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
                  "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.", "10"),
])

fails = []
for name, system in (("zero_shot", None), ("few_shot", SYS)):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user_content(Q)}]
    ref = tpl.render(messages=msgs, add_generation_prompt=True,
                     bos_token=tok.bos_token, eos_token=tok.eos_token)
    ours = render_prompt(Q, system)
    ok = ref == ours
    print(f"{name}: jinja == prompting.py -> {ok}")
    if not ok:
        fails.append(name)
        print(json.dumps({"ref": ref[-300:], "ours": ours[-300:]}, indent=2))

# target terminator
t = render_target("Natalia sold 48/2 = 24 clips in May.\nANSWER: 72")
ids = tok(t, add_special_tokens=False)["input_ids"]
print("target last token id:", ids[-1], tok.convert_ids_to_tokens(ids[-1:]))
if ids[-1] != 106:
    fails.append("stop_token")

# a full training row round-trips through the tokenizer with a single bos
p = render_prompt(Q, None)
pids = tok(p, add_special_tokens=False)["input_ids"]
print("prompt first tokens:", tok.convert_ids_to_tokens(pids[:4]))
if pids.count(2) != 1:
    fails.append("bos_count")

print("FAILS:", fails or "none")
sys.exit(1 if fails else 0)
