"""Shared formatting: reproduce the grader's rendering byte-for-byte.

The grader (evaluate.py -> inspect_evals/gsm8k) does:
  system_message(<10 fewshot exemplars>)  +  prompt_template(MATH_PROMPT_TEMPLATE)
and renders the conversation with templates/gemma3.jinja through vLLM.
Everything here is derived from those two files so training and grading agree.
"""
from __future__ import annotations

import hashlib
import os

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
EOT = "<end_of_turn>"


def template_sha256() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fewshot_system_message() -> str:
    """The exact system message the grader builds (fewshot=10, seed=42, shuffled)."""
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
    return "\n\n".join(sample_to_fewshot(s) for s in fewshots)


def user_content(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def render_prompt(question: str, system: str | None) -> str:
    """Manual render of gemma3.jinja up to and including '<start_of_turn>model\\n'."""
    prefix = (system.strip() + "\n\n") if system else ""
    u = user_content(question).strip()
    return f"{BOS}<start_of_turn>user\n{prefix}{u}{EOT}\n<start_of_turn>model\n"


def render_target(solution: str) -> str:
    """The assistant turn the model must produce, terminated by the grader's stop token.

    Idempotent: the data files already carry the stop token so that
    `awm exp_protocol` can verify it, and this must not double it.
    """
    s = solution.strip()
    return s if s.endswith(EOT) else s + EOT


def check_against_jinja(question: str, system: str | None) -> tuple[str, str]:
    """Render the same conversation through the actual jinja file; must match render_prompt."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    with open(TEMPLATE_PATH) as f:
        chat_template = f.read()
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_content(question)})
    ref = tok.apply_chat_template(
        msgs, chat_template=chat_template, tokenize=False, add_generation_prompt=True
    )
    return render_prompt(question, system), ref
