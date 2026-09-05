#!/usr/bin/env python3
"""The single definition of how a (prompt, completion) pair becomes training text.

Everything that touches training data imports `render` from here, so training and
grading cannot drift apart (pitfall: template_unreachable). The grader's template
is read from templates/gemma3.jinja and its sha256 is asserted, so an edit to the
template breaks the build instead of silently changing the training format.
"""
from __future__ import annotations

import hashlib
import os

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(ROOT, "templates/gemma3.jinja")
TEMPLATE_SHA256 = "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"
STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "

_raw = open(TEMPLATE_PATH, "rb").read()
_got = hashlib.sha256(_raw).hexdigest()
assert _got == TEMPLATE_SHA256, f"templates/gemma3.jinja changed: {_got}"
TEMPLATE = _raw.decode()

_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(os.environ["PTB_BASE_MODEL_SNAPSHOT"])
    return _tok


def render(prompt: str, completion: str) -> tuple[str, str]:
    """Return (prefix, full).

    prefix is everything the grader will have put in the context before the model
    starts generating; full is prefix + the supervised target, terminated by the
    token vLLM stops on. Loss is taken on full[len(prefix):] only.
    """
    prefix = tokenizer().apply_chat_template(
        [{"role": "user", "content": prompt}],
        chat_template=TEMPLATE,
        tokenize=False,
        add_generation_prompt=True,
    )
    c = completion.strip()
    if not c.endswith(STOP_TOKEN):
        c += STOP_TOKEN
    return prefix, prefix + c


if __name__ == "__main__":
    p, f = render("What is 2+2?\n\nReasoning:", "2 + 2 = 4\n\nANSWER: 4")
    print(repr(p))
    print(repr(f))
    t = tokenizer()
    ids = t(f, add_special_tokens=False)["input_ids"]
    print("last 5 ids", ids[-5:], t.convert_ids_to_tokens(ids[-5:]))
    print("first 3 ids", ids[:3], t.convert_ids_to_tokens(ids[:3]))
