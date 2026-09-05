"""Rendering that is byte-for-byte the grader's.

evaluate.py starts `vllm serve <model> --chat-template templates/gemma3.jinja`
and talks to /v1/chat/completions, so the server renders the conversation with
THAT jinja file, not with whatever the tokenizer ships. This module renders with
the same file (hash-checked) so training strings and grading strings agree.

Pitfall addressed: template_unreachable (pitfalls.yaml).
"""

from __future__ import annotations

import hashlib
import os

from jinja2 import Environment
from jinja2.exceptions import TemplateError

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

BOS = "<bos>"
END_OF_TURN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def template_sha256() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _env() -> Environment:
    env = Environment(trim_blocks=False, lstrip_blocks=False)

    def raise_exception(msg):  # noqa: ANN001
        raise TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    return env


def render_jinja(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Render exactly as the vLLM server does with templates/gemma3.jinja."""
    with open(TEMPLATE_PATH) as f:
        tpl = _env().from_string(f.read())
    return tpl.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        bos_token=BOS,
    )


def render_prompt_fast(system: str | None, user: str) -> str:
    """Hand-rolled equivalent of render_jinja([system?, user], add_generation_prompt=True).

    Kept separate so build_data.py can assert the two agree on every row shape.
    """
    prefix = (system.strip() + "\n\n") if system else ""
    return (
        f"{BOS}<start_of_turn>user\n{prefix}{user.strip()}{END_OF_TURN}\n"
        f"<start_of_turn>model\n"
    )


def render_target(assistant: str) -> str:
    """What the model must generate: the trimmed content, then the stop token.

    The template writes '<end_of_turn>\\n' after assistant content, but the
    trailing newline belongs to the next turn's framing and is never generated:
    vLLM stops the moment token 106 (<end_of_turn>) appears.
    """
    return f"{assistant.strip()}{END_OF_TURN}"


def self_check() -> None:
    """Assert render_prompt_fast == render_jinja for both shapes."""
    sys_msg = "few shot A\n\nfew shot B"
    user = "Solve this.\n\nReasoning:"
    for system in (None, sys_msg):
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        a = render_jinja(msgs)
        b = render_prompt_fast(system, user)
        assert a == b, f"MISMATCH\njinja={a!r}\nfast ={b!r}"
    print("fmt.self_check ok; template sha256:", template_sha256())


if __name__ == "__main__":
    self_check()
