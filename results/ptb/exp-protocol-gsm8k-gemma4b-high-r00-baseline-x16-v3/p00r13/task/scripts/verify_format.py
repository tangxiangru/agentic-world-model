#!/usr/bin/env python3
"""Guard against pitfalls `template_unreachable`, `eos_mismatch`, `double_answer_format`.

1. Render one conversation with templates/gemma3.jinja (the file evaluate.py hands
   to vLLM) and compare byte-for-byte with build_sft_data.render_prompt.
2. Check the completion's last token is <end_of_turn> (id 106) and that it is an
   eos id in the base generation_config.
3. Run the grader's own scorer over a synthetic completion in our target format.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft_data import MATH_PROMPT_TEMPLATE, render_prompt  # noqa: E402

TASK = Path(__file__).resolve().parent.parent
SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TPL = TASK / "templates" / "gemma3.jinja"

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(SNAP)
tpl = TPL.read_text()
print("template sha256:", hashlib.sha256(tpl.encode()).hexdigest()[:16])

SYSTEM = "Q1\n\nReasoning:\nr1\n\nANSWER: 5\n\nQ2\n\nReasoning:\nr2\n\nANSWER: 7"
QUESTION = "Sam has 3 apples and buys 4 more. How many does he have?"
BODY = "Sam starts with 3 apples.\nHe buys 4 more, so 3 + 4 = 7.\n\nANSWER: 7"

for system in (None, SYSTEM):
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=QUESTION)}
    ]
    ref = tok.apply_chat_template(msgs, chat_template=tpl, tokenize=False,
                                  add_generation_prompt=True)
    ours = render_prompt(system, QUESTION)
    tag = "with-system" if system else "no-system"
    assert ref == ours, (tag, repr(ref[:200]), repr(ours[:200]), repr(ref[-200:]), repr(ours[-200:]))
    print(f"prompt render {tag}: IDENTICAL ({len(tok(ours, add_special_tokens=False)['input_ids'])} tok)")

    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": BODY}],
                                   chat_template=tpl, tokenize=False)
    # the grader never renders the assistant turn, but our training row must be a
    # prefix-consistent split of it
    assert full.startswith(ours[: -len("<start_of_turn>model\n")]), "assistant render diverges"

comp = BODY + "<end_of_turn>"
ids = tok(comp, add_special_tokens=False)["input_ids"]
print("completion last 3 ids:", ids[-3:], "->", tok.convert_ids_to_tokens(ids[-3:]))
assert ids[-1] == 106, ids[-1]
gen_cfg = json.loads((Path(SNAP) / "generation_config.json").read_text())
assert 106 in gen_cfg["eos_token_id"], gen_cfg
print("stop token <end_of_turn>=106 is in generation_config eos_token_id:", gen_cfg["eos_token_id"])

# 3. the actual scorer
from inspect_ai.scorer._common import match_str  # noqa: E402

for target, ok in [("7", True), ("8", False)]:
    _, matched = match_str(BODY, target, location="end", numeric=True)
    assert matched == ok, (target, matched)
print("inspect match(location='end', numeric=True) reads our final line correctly")

# 4. sanity over a built file, if given
if len(sys.argv) > 1:
    path = Path(sys.argv[1])
    n, bad_stop, bad_marker, lens = 0, 0, 0, []
    for line in path.open():
        r = json.loads(line)
        n += 1
        if not r["completion"].endswith("<end_of_turn>"):
            bad_stop += 1
        if r["completion"].count("ANSWER:") != 1:
            bad_marker += 1
        if n <= 3000:
            lens.append(len(tok(r["prompt"] + r["completion"], add_special_tokens=False)["input_ids"]))
    lens.sort()
    print(f"{path}: n={n} bad_stop={bad_stop} bad_answer_marker={bad_marker}")
    print(f"  token len p50={lens[len(lens)//2]} p95={lens[int(len(lens)*.95)]} max={lens[-1]} (first {len(lens)} rows)")
print("ALL FORMAT CHECKS PASSED")
