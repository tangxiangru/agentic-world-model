"""Guard against the template_unreachable pitfall.

Asserts (a) the prompt template copied into scripts/fmt.py is still identical to
inspect_evals' own, and (b) our renderer reproduces, byte-for-byte, what the tokenizer
produces with the grader's templates/gemma3.jinja - zero-shot and with a system message.
Run before every training launch.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def main() -> int:
    assert MATH_PROMPT_TEMPLATE == fmt.MATH_PROMPT_TEMPLATE, "prompt template drifted"
    tok = AutoTokenizer.from_pretrained(fmt.BASE_MODEL)
    tok.chat_template = fmt.template_text()
    q = "Q?"
    sysmsg = "SYS-A\n\nSYS-B"
    pairs = [
        (
            tok.apply_chat_template(
                [{"role": "user", "content": fmt.user_content(q)}],
                tokenize=False,
                add_generation_prompt=True,
            ),
            fmt.render_prompt(q),
        ),
        (
            tok.apply_chat_template(
                [
                    {"role": "system", "content": sysmsg},
                    {"role": "user", "content": fmt.user_content(q)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            ),
            fmt.render_prompt(q, fewshot_system=sysmsg),
        ),
    ]
    for a, b in pairs:
        assert a == b, f"render mismatch:\n{a!r}\n{b!r}"
    comp = fmt.render_completion("Step 1. 2+2 = 4.", "4")
    ids = tok(comp, add_special_tokens=False).input_ids
    assert ids[-1] == tok.convert_tokens_to_ids(fmt.STOP_TOKEN) == 106, ids[-4:]
    assert comp.count(fmt.ANSWER_MARKER) == 1
    print(f"OK template sha {fmt.template_sha()} stop_token {fmt.STOP_TOKEN} id {ids[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
