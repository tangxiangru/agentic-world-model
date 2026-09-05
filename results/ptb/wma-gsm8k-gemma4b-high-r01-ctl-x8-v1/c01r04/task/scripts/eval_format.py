"""Single source of truth for the exact strings the grader renders.
Mirrors inspect_evals/gsm8k/gsm8k.py + templates/gemma3.jinja byte-for-byte."""
import os, hashlib
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"

def gemma_template():
    with open(TEMPLATE_PATH) as f:
        t = f.read()
    return t, hashlib.sha256(t.encode()).hexdigest()

def _record_to_sample(rec):
    DELIM = "####"
    answer = rec["answer"].split(DELIM)
    target = answer.pop().strip()
    reasoning = DELIM.join(answer).strip()
    return rec["question"], reasoning, target

def fewshot_prefix(n=10, seed=42):
    """Reproduces inspect's system_message: hf_dataset(train, shuffle=True, seed=42, limit=n)."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(n))
    parts = []
    for rec in ds:
        q, reasoning, target = _record_to_sample(rec)
        parts.append(f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}")
    return "\n\n".join(parts)

def fewshot_questions(n=10, seed=42):
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed).select(range(n))
    return [rec["question"].strip() for rec in ds]

def user_content(question, prefix=None):
    """Content of the single user turn as gemma3.jinja renders it (system folded in)."""
    body = MATH_PROMPT_TEMPLATE.format(prompt=question)
    if prefix:
        return (prefix + "\n\n" + body).strip()
    return body.strip()

def render_prompt(question, prefix=None):
    """Full rendered prompt string WITHOUT bos (tokenizer adds bos)."""
    return "<start_of_turn>user\n" + user_content(question, prefix) + "<end_of_turn>\n<start_of_turn>model\n"

def render_target(solution_text, answer):
    """Assistant turn: content is `| trim`-ed by the template, then <end_of_turn>."""
    body = (solution_text.strip() + "\n\nANSWER: " + str(answer).strip()).strip()
    return body + "<end_of_turn>\n"
