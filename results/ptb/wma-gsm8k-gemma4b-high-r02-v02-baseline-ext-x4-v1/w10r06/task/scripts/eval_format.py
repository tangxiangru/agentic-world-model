"""Byte-exact reproduction of the grader's prompt rendering.

The grader (evaluate.py -> inspect_evals/gsm8k) builds, for every test item:
    messages = [system(10-shot text), user(MATH_PROMPT_TEMPLATE.format(prompt=question))]
and renders them with templates/gemma3.jinja. Anything we train on must be
rendered by the same jinja file (pitfalls.yaml:template_unreachable).
"""
from __future__ import annotations

import os

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(TASK_DIR, "templates", "gemma3.jinja")

# verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

DELIM = "####"


def read_template() -> str:
    with open(TEMPLATE_PATH) as f:
        return f.read()


def record_to_sample(record):
    """inspect_evals.gsm8k.record_to_sample, minus the id."""
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    reasoning = DELIM.join(answer)
    return record["question"], target, reasoning.strip()


def sample_to_fewshot(question: str, target: str, reasoning: str) -> str:
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def build_fewshot_system(n: int = 10, seed: int = 42) -> str:
    """Exactly what inspect's hf_dataset(shuffle=True, seed=42, limit=n) yields."""
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample as r2s, sample_to_fewshot as s2f

    fewshots = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=r2s,
        shuffle=True,
        seed=seed,
        limit=n,
    )
    return "\n\n".join([s2f(s) for s in fewshots])


def user_content(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def render_prompt(tokenizer, question: str, system: str | None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_content(question)})
    return tokenizer.apply_chat_template(
        msgs, chat_template=read_template(), tokenize=False, add_generation_prompt=True
    )
