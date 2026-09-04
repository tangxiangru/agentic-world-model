"""Shared rendering helpers: one place where the grader's contract is encoded."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = ROOT / "templates" / "gemma3.jinja"
BASE_SNAPSHOT = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
                 "cc012e0a6d0787b4adcc0fa2c4da74402494554d")

# the terminator templates/gemma3.jinja writes after every turn; token id 106,
# which is in the base generation_config's eos_token_id list, so vLLM stops on it
STOP_TOKEN = "<end_of_turn>"
ANSWER_MARKER = "ANSWER: "

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def template_text() -> str:
    return TEMPLATE_PATH.read_text()


def template_sha() -> str:
    return hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest()[:16]


def load_tokenizer(path: str = BASE_SNAPSHOT):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    tok.chat_template = template_text()   # the grader's template, verbatim
    return tok


def render_prompt(tok, messages) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
