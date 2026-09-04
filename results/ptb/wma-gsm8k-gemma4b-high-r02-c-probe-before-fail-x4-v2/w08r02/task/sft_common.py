#!/usr/bin/env python3
"""Shared rendering + tokenisation for the SFT runs.

The one thing this file exists to guarantee is that the string the trainer sees
is the string the grader will send. `render_prompt` renders through the very
same jinja file evaluate.py hands to vLLM (`templates/gemma3.jinja`), and
`TEMPLATE_SHA256` is checked at import so a silent edit cannot go unnoticed
(pitfall: template_unreachable).
"""
from __future__ import annotations

import hashlib
import json
import os

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")
BASE_SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)

with open(TEMPLATE_PATH, "rb") as _f:
    TEMPLATE_BYTES = _f.read()
TEMPLATE_SHA256 = hashlib.sha256(TEMPLATE_BYTES).hexdigest()
CHAT_TEMPLATE = TEMPLATE_BYTES.decode("utf-8")

END_OF_TURN = "<end_of_turn>"


def get_tokenizer(path: str = BASE_SNAPSHOT):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    tok.chat_template = CHAT_TEMPLATE
    return tok


def render_prompt(tok, system: str | None, user: str) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode_row(tok, row: dict, max_seq_len: int) -> dict | None:
    """input_ids = prompt + target + <end_of_turn>; loss only on the target part.

    vLLM's chat endpoint tokenises the templated string with
    add_special_tokens=False (ChatCompletionRequest.add_special_tokens defaults
    to False), and the template itself emits <bos>. We do the same, so there is
    exactly one BOS in both training and serving.
    """
    prompt = render_prompt(tok, row.get("system"), row["user"])
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    assert row["target"].endswith(END_OF_TURN), "target must carry the terminator"
    t_ids = tok(row["target"], add_special_tokens=False)["input_ids"]
    ids = p_ids + t_ids
    if len(ids) > max_seq_len:
        return None
    labels = [-100] * len(p_ids) + list(t_ids)
    return {"input_ids": ids, "labels": labels, "n_prompt": len(p_ids), "n_target": len(t_ids)}


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]
