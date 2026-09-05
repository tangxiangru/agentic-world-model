"""Exact reproduction of the grader's prompt rendering.

Everything that turns a (question, solution) pair into training text lives here so
training and grading cannot drift apart (pitfall: template_unreachable).

Sources of truth, read at import time and hashed:
  - /home/ben/task/templates/gemma3.jinja        (evaluate.py passes this to vLLM)
  - inspect_evals.gsm8k.MATH_PROMPT_TEMPLATE     (the user turn)
  - inspect_evals.gsm8k.sample_to_fewshot        (the 10-shot system message)
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"
SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)

# byte-for-byte copy of inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "


def template_sha() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def assert_prompt_template_matches() -> None:
    """Fail loudly if the installed grader's template is not the one copied above."""
    from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE as LIVE

    assert LIVE == MATH_PROMPT_TEMPLATE, "MATH_PROMPT_TEMPLATE drifted from the grader"


@lru_cache(maxsize=1)
def fewshot_system_message() -> str:
    """The exact system message evaluate.py builds: 10 train items, shuffled seed 42."""
    import datasets

    ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=42).select(range(10))
    blocks = []
    for rec in ds:
        answer = rec["answer"].split("####")
        target = answer.pop().strip()
        reasoning = "####".join(answer).strip()
        blocks.append(f"{rec['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}")
    return "\n\n".join(blocks)


def user_turn(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def render_prompt(question: str, fewshot: bool) -> str:
    """Render exactly what the model sees at generation time, gemma3.jinja applied by hand.

    The template puts the system content inside the first user turn followed by a
    blank line, trims each message body, and opens '<start_of_turn>model\\n'.
    """
    body = user_turn(question).strip()
    if fewshot:
        body = fewshot_system_message().strip() + "\n\n" + body
    return "<bos><start_of_turn>user\n" + body + "<end_of_turn>\n<start_of_turn>model\n"


def render_target(solution: str, answer: str) -> str:
    """The completion: reasoning, the single answer marker, then the stop token.

    The grader is match(numeric=True, location='end'), so the very last thing in the
    completion has to be the number.
    """
    return solution.strip() + f"\n\n{ANSWER_MARKER}{answer.strip()}" + STOP_TOKEN + "\n"


def verify_against_jinja(question: str) -> tuple[str, str]:
    """Render one prompt both ways (hand-rolled vs the real jinja template)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    with open(TEMPLATE_PATH) as f:
        chat_template = f.read()
    messages = [
        {"role": "system", "content": fewshot_system_message()},
        {"role": "user", "content": user_turn(question)},
    ]
    ref = tok.apply_chat_template(
        messages, chat_template=chat_template, tokenize=False, add_generation_prompt=True
    )
    return render_prompt(question, fewshot=True), ref


if __name__ == "__main__":
    assert_prompt_template_matches()
    q = "Natalia sold clips to 48 of her friends in April. How many in total?"
    mine, ref = verify_against_jinja(q)
    print("template sha256[:12]:", template_sha())
    print("hand-rolled == jinja:", mine == ref)
    if mine != ref:
        for i, (a, b) in enumerate(zip(mine, ref)):
            if a != b:
                print("first diff at", i)
                print("mine:", repr(mine[max(0, i - 80) : i + 80]))
                print("ref :", repr(ref[max(0, i - 80) : i + 80]))
                break
        print("len mine", len(mine), "len ref", len(ref))
    print("---- zero-shot prompt ----")
    print(repr(render_prompt(q, fewshot=False)))
    print("---- target ----")
    print(repr(render_target("2 + 2 = 4.", "4")))
