"""Reproduce, byte-for-byte, the prompt the grader (evaluate.py -> inspect_evals/gsm8k)
sends to the model, and the gemma3.jinja rendering the vLLM provider applies.

Sources:
  /usr/local/lib/python3.10/dist-packages/inspect_evals/gsm8k/gsm8k.py
  /home/ben/task/templates/gemma3.jinja
"""
from __future__ import annotations

import hashlib
import os

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"
TEMPLATE_SHA256 = "e0e5e05b3f8c1e2a"  # filled by check_template_hash()


def user_prompt(question: str) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=question)


def fewshot_block(question: str, reasoning: str, answer: str) -> str:
    """inspect_evals.gsm8k.sample_to_fewshot"""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def render(system: str | None, user: str, add_generation_prompt: bool = True) -> str:
    """Equivalent of templates/gemma3.jinja for a single user turn."""
    out = "<bos>"
    prefix = (system.strip() + "\n\n") if system else ""
    out += "<start_of_turn>user\n" + prefix + user.strip() + "<end_of_turn>\n"
    if add_generation_prompt:
        out += "<start_of_turn>model\n"
    return out


def gsm8k_fewshot_system(n: int = 10, seed: int = 42) -> str:
    """The exact system message inspect_evals/gsm8k builds with fewshot=10, seed=42.

    hf_dataset(..., shuffle=True, seed=42, limit=10) -> Dataset.shuffle(seed).select(range(10))
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=seed)
    blocks = []
    for rec in ds.select(range(n)):
        q = rec["question"]
        parts = rec["answer"].split("####")
        target = parts[-1].strip()
        reasoning = "####".join(parts[:-1]).strip()
        blocks.append(fewshot_block(q, reasoning, target))
    return "\n\n".join(blocks)


def check_template_hash() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


if __name__ == "__main__":
    from transformers import AutoTokenizer

    S = os.environ.get(
        "BASE_MODEL",
        "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
    )
    tok = AutoTokenizer.from_pretrained(S)
    sysmsg = gsm8k_fewshot_system()
    q = "Janet has 3 apples and buys 5 more. How many does she have?"
    mine = render(sysmsg, user_prompt(q))

    # cross-check against the actual jinja template as vLLM would apply it
    with open(TEMPLATE_PATH) as f:
        tmpl = f.read()
    tok.chat_template = tmpl
    theirs = tok.apply_chat_template(
        [{"role": "system", "content": sysmsg}, {"role": "user", "content": user_prompt(q)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    print("template sha256:", check_template_hash())
    print("MATCH:", mine == theirs)
    if mine != theirs:
        for i, (a, b) in enumerate(zip(mine, theirs)):
            if a != b:
                print("first diff at", i, repr(mine[i - 40 : i + 40]), "|||", repr(theirs[i - 40 : i + 40]))
                break
        print(len(mine), len(theirs))
    print("--- rendered tail ---")
    print(repr(mine[-600:]))
    print("system tokens:", len(tok(sysmsg, add_special_tokens=False)["input_ids"]))
    print("full prompt tokens:", len(tok(mine, add_special_tokens=False)["input_ids"]))
    print("zero-shot prompt tokens:", len(tok(render(None, user_prompt(q)), add_special_tokens=False)["input_ids"]))
