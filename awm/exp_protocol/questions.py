"""Nothing is guessed: what the card does not settle becomes a question to the scientist.

The order is the order a scientist would answer them in; ``check`` prints
them in this order so the first question is always the most basic one.
"""

from __future__ import annotations

from typing import Any

from .schema import TRAINING_FAMILIES, get

REQUIRED: dict[str, str] = {
    "situation.elapsed_h": "How many hours into the run are you (bash timer.sh)?",
    "situation.trigger": "What observation led to this experiment? One line, pointing at a file if you can.",
    "problem.statement": "What is going wrong in the current model, concretely? One or two sentences.",
    "hypothesis.claim": "What do you expect this run to change, and against what (base model or a previous card)? One sentence.",
    "setup.parent_checkpoint.path": "Which checkpoint does the run start from (a local directory or the base model id)?",
    "setup.parent_checkpoint.origin": "What produced that checkpoint: base_model, or which card id?",
    "setup.data": "Which training data file(s) does the run read (paths), where did each come from, and how many examples? (asked only when the method family trains on target text; leave [] for decode-config, merge and eval-only cards)",
    "setup.method.family": "Which method family: sft | rft | dpo | grpo | distill | merge | decode-config | other?",
    "setup.command.argv": "What is the exact launch command (full argv, as you will run it)?",
    "setup.command.cwd": "From which directory will you run it?",
    "setup.output_dir": "Where will the trainer save checkpoints (absolute path inside your task dir)?",
    "setup.checkpoints.keep": "Which checkpoints will you keep: all | last | best | <n>? (later cards may start from them)",
    "evaluation.protocol.n": "How many benchmark items does each evaluation use (evaluate.py --limit N)?",
}


#: Asked only when setup.method.family trains on target text (schema.TRAINING_FAMILIES).
TRAINING_REQUIRED: dict[str, str] = {
    "setup.method.stop_token": "Which stop token does the grading chat template end a turn with (e.g. <|im_end|>)? Every training target must end with it.",
    "setup.method.hyperparams.max_seq_len": "What max_seq_len will the trainer use? (preflight estimates how many rows would truncate)",
}


def _missing(card: dict[str, Any], dotted: str) -> bool:
    value = get(card, dotted)
    return value is None or value == [] or value == ""


def missing_fields(card: dict[str, Any]) -> list[tuple[str, str]]:
    family = get(card, "setup.method.family")
    trains = family is None or family in TRAINING_FAMILIES
    asked = [(f, q) for f, q in REQUIRED.items()
             if _missing(card, f) and (trains or f != "setup.data")]
    if get(card, "setup.method.family") in TRAINING_FAMILIES:
        asked += [(f, q) for f, q in TRAINING_REQUIRED.items() if _missing(card, f)]
    return asked
