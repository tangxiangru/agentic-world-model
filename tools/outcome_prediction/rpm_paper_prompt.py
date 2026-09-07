"""Figure 7 inference-only RPM prompt and an explicit immediate-target adaptation.

Attribution: Thomas Simon Foster et al., *AI Research Preference Models*,
arXiv:2608.13940v2 (2026), Figure 7, https://arxiv.org/html/2608.13940v2.
The source is licensed CC BY 4.0: https://creativecommons.org/licenses/by/4.0/.
The eventual template below reproduces its wording, normalizing PDF line wraps
and the page break. The immediate template changes the target and the evaluation
criteria that otherwise reward hypothetical future upgrades or bug fixes.

This module only formats already-audited inputs. It neither establishes code
provenance nor removes outcome leakage. Callers must supply code available before
execution, exclude candidate outcomes, and establish an appropriate history cutoff.
The paper's eventual target is the best measured score in a candidate's subtree;
it must not be scored against an immediate-checkpoint label.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

PAPER_URL = "https://arxiv.org/html/2608.13940v2"
PAPER_LICENSE = "https://creativecommons.org/licenses/by/4.0/"

EVENTUAL_TEMPLATE = r"""You are a principal investigator allocating compute budget to one of two branches. Decide which branch is more likely to yield the best eventual test score after several iterations. Emphasize extensibility, fixability, and promise relative to what has already been tried.

Task description:
{task_desc}

Context from other solutions and their scores (not the candidates). Use this to identify promising gaps and avoid redundant directions:
{context_text}

Candidate A -- Plan:
{plan_A}
Candidate A -- Code:
{code_A}

Candidate B -- Plan:
{plan_B}
Candidate B -- Code:
{code_B}

Step-by-step evaluation:

Problem-model fit:

Does each candidate's formulation and objective align with the task? Any risks of leakage or misalignment?
Note which issues are trivially fixable vs. fundamental.

Extensibility and upgrade path:

How modular is the code? How straightforward is it to add stronger models, features, or training strategies in 1 to 3 iterations?

Learning curve projection:

Based on current choices, estimate how performance might improve over the next few iterations. Identify low-hanging fruit (data cleaning, features, hyperparameters, regularization, architecture changes).

Context-informed novelty:

Relative to {context_text}, does the candidate explore a fresh, promising region or iterate intelligently on a proven one? Avoid branches that mirror underperforming context without new leverage.

Risk-adjusted potential:

Balance upside (ceiling) against effort/risk to realize it. Bugs are acceptable if the approach is sound and fixes are clear; penalize only for hard-to-remedy conceptual flaws.

Decision policy:

Choose the candidate whose direction offers higher expected long-term best test score and a credible path to get there.

Output format (STRICT):

Provide reasoning following the steps above.

End with a single final answer: A for Candidate A or B for Candidate B.

Provide your answer inside a \boxed{A} or \boxed{B}."""

# Explicit substitutions leave the rest of Figure 7, including boxed output and
# the repeated context placeholder, unchanged. No rationale-length cap is added.
IMMEDIATE_SUBSTITUTIONS = (
    (
        (
            "Decide which branch is more likely to yield the best eventual test score after several "
            "iterations. Emphasize extensibility, fixability, and promise relative to what has already "
            "been tried."
        ),
        (
            "Decide which candidate is more likely to yield the better immediate test score after "
            "the supplied plan and code are executed. Emphasize the implemented experiment and "
            "evidence from what has already been tried, not possible future descendants or upgrades."
        ),
    ),
    (
        "Note which issues are trivially fixable vs. fundamental.",
        "Note which issues affect this execution; do not assume any unimplemented fixes.",
    ),
    ("Extensibility and upgrade path:", "Implementation and execution:"),
    (
        (
            "How modular is the code? How straightforward is it to add stronger models, features, "
            "or training strategies in 1 to 3 iterations?"
        ),
        (
            "Does the supplied code implement the plan correctly? Evaluate the models, features, "
            "and training strategies actually supplied, without adding future upgrades."
        ),
    ),
    (
        (
            "Based on current choices, estimate how performance might improve over the next few "
            "iterations. Identify low-hanging fruit (data cleaning, features, hyperparameters, "
            "regularization, architecture changes)."
        ),
        (
            "Based on current choices and the specified training budget, estimate the immediate "
            "test score. Consider the supplied data cleaning, features, hyperparameters, "
            "regularization, and architecture, without assuming additional research iterations."
        ),
    ),
    (
        (
            "Balance upside (ceiling) against effort/risk to realize it. Bugs are acceptable if "
            "the approach is sound and fixes are clear; penalize only for hard-to-remedy "
            "conceptual flaws."
        ),
        (
            "Balance expected immediate performance against execution risk. Evaluate the code "
            "as supplied: penalize bugs that affect this execution, even if fixes are clear; "
            "do not assume future repairs."
        ),
    ),
    (
        (
            "Choose the candidate whose direction offers higher expected long-term best test "
            "score and a credible path to get there."
        ),
        (
            "Choose the candidate with the higher expected immediate test score from the "
            "supplied experiment, not the best possible future descendant."
        ),
    ),
)

IMMEDIATE_TEMPLATE = EVENTUAL_TEMPLATE
for _old, _new in IMMEDIATE_SUBSTITUTIONS:
    if IMMEDIATE_TEMPLATE.count(_old) != 1:
        raise RuntimeError("Immediate-target substitution does not uniquely match Figure 7")
    IMMEDIATE_TEMPLATE = IMMEDIATE_TEMPLATE.replace(_old, _new, 1)

_PLAN_KEYS = ("earliest_plan", "full_plan", "plan", "plan_time_setup", "setup")
_CODE_KEYS = ("earliest_code", "code", "code_files", "launch_scripts", "source_code")
_PLACEHOLDERS = re.compile(r"\{(task_desc|context_text|plan_A|code_A|plan_B|code_B)\}")


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def _candidate_parts(candidate: Any) -> tuple[str, str]:
    """Preserve full supplied material, including additional candidate metadata.

    Recognized aliases are not claims of provenance. If multiple representations
    are supplied, retain all of them rather than silently choosing or truncating.
    Additional candidate information stays in its plan section, not in the
    historical context that Figure 7 describes as excluding the candidates.
    """
    if not isinstance(candidate, Mapping):
        return _render(candidate), "Code not supplied."
    plans = {key: candidate[key] for key in _PLAN_KEYS if key in candidate}
    codes = {key: candidate[key] for key in _CODE_KEYS if key in candidate}
    remaining = {key: value for key, value in candidate.items() if key not in plans | codes}
    plan = next(iter(plans.values())) if len(plans) == 1 else plans
    if remaining:
        plan = {"plan": plan, "additional_candidate_information": remaining} if plans else remaining
    code = next(iter(codes.values())) if len(codes) == 1 else codes
    return (
        _render(plan) if plans or remaining else "Plan not supplied.",
        _render(code) if codes else "Code not supplied.",
    )


def render_prompt(payload: Mapping[str, Any], target: str = "immediate") -> str:
    """Render Figure 7 without truncating, redacting, or recursively templating inputs.

    Required keys: candidate_A and candidate_B. Candidate values may be full plan
    strings or mappings with the plan/code aliases above. The first supplied
    task_desc/task_description/task field supplies the task. All other top-level
    fields, including measured history and parent checkpoints, become context.
    Input mutation, temporal filtering, provenance checks, and A/B randomization
    belong to the caller. No scores or other fields are silently discarded.
    """
    if target not in {"immediate", "eventual"}:
        raise ValueError("target must be 'immediate' or 'eventual'")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    if "candidate_A" not in payload or "candidate_B" not in payload:
        raise ValueError("Both candidate_A and candidate_B are required")
    context = {
        key: value for key, value in payload.items() if key not in {"candidate_A", "candidate_B"}
    }
    task = "Compare the supplied candidate experiments using the stated evaluation metric."
    for key in ("task_desc", "task_description", "task"):
        if key in context:
            task = context.pop(key)
            break
    plan_a, code_a = _candidate_parts(payload["candidate_A"])
    plan_b, code_b = _candidate_parts(payload["candidate_B"])
    values = {
        "task_desc": _render(task),
        "context_text": _render(context),
        "plan_A": plan_a,
        "code_A": code_a,
        "plan_B": plan_b,
        "code_B": code_b,
    }
    template = IMMEDIATE_TEMPLATE if target == "immediate" else EVENTUAL_TEMPLATE
    return _PLACEHOLDERS.sub(lambda match: values[match.group(1)], template)


def _mask_markdown_code(text: str) -> str:
    """Blank Markdown code spans/fences without changing character offsets.

    Literal LaTeX in quoted code is evidence being discussed, not an answer.
    Unclosed fenced blocks extend to EOF; unmatched inline backticks are prose.
    """
    masked = list(text)

    def blank(start: int, end: int) -> None:
        for index in range(start, end):
            if masked[index] not in "\r\n":
                masked[index] = " "

    fence = None
    offset = 0
    for line in text.splitlines(keepends=True):
        if fence is not None:
            blank(offset, offset + len(line))
            character, minimum_length = fence
            if re.fullmatch(
                r" {0,3}" + re.escape(character) + "{" + str(minimum_length) + r",}[ \t\r\n]*",
                line,
            ):
                fence = None
        else:
            opening = re.match(r" {0,3}(`{3,}|~{3,})([^\r\n]*)", line)
            if opening and not (opening[1][0] == "`" and "`" in opening[2]):
                fence = (opening[1][0], len(opening[1]))
                blank(offset, offset + len(line))
        offset += len(line)

    fenced_masked = "".join(masked)
    runs = list(re.finditer(r"`+", fenced_masked))
    index = 0
    while index < len(runs):
        opening = runs[index]
        prefix = fenced_masked[: opening.start()]
        if (len(prefix) - len(prefix.rstrip("\\"))) % 2:
            index += 1
            continue
        closing_index = next(
            (j for j in range(index + 1, len(runs)) if runs[j].group() == opening.group()),
            None,
        )
        if closing_index is None:
            index += 1
            continue
        blank(opening.start(), runs[closing_index].end())
        index = closing_index + 1
    return "".join(masked)


def decode_boxed_response(raw: Mapping[str, Any], swapped: bool = False) -> dict[str, Any]:
    """Decode exactly one final boxed A/B from a successful Claude CLI response.

    `choice_a` is mapped back to canonical candidate A; `displayed_choice` records
    the answer as seen by the model. This is a forced-choice output, not a supplied
    probability, so no artificial p_a/confidence is manufactured. The entire
    reasoning is retained. Failed, absent, repeated, and ambiguous boxes raise
    ValueError. Literal examples inside Markdown inline/fenced code are ignored.
    A final box may be wrapped in normal LaTeX math delimiters.
    """
    if not isinstance(raw, Mapping) or raw.get("is_error") or raw.get("subtype") != "success":
        raise ValueError("CLI did not return a successful result")
    answer = raw.get("result")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("CLI result must contain nonempty text")
    if not isinstance(swapped, bool):
        raise TypeError("swapped must be a bool")
    unquoted = _mask_markdown_code(answer)
    boxes = list(re.finditer(r"\\boxed\s*\{([^{}]*)\}", unquoted))
    # Count all unquoted commands, including malformed/nested boxes, so a valid
    # final answer cannot conceal an earlier conflicting unparseable answer.
    if len(boxes) != 1 or len(re.findall(r"\\boxed\b", unquoted)) != 1:
        raise ValueError("Expected exactly one unambiguous boxed answer")
    box = boxes[0]
    choice = box.group(1).strip()
    if choice not in {"A", "B"}:
        raise ValueError("Boxed answer must be A or B")
    if not re.fullmatch(r"(?:\s|\$|\\\)|\\\]|[.!])*", answer[box.end() :]):
        raise ValueError("Boxed answer must be final")
    return {
        "choice_a": (choice == "A") != swapped,
        "displayed_choice": choice,
        "rationale": answer,
    }
