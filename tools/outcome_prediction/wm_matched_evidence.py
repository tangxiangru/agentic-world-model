"""Positive-only, prospective treatment declarations for matched WM evidence.

RFT in this frozen cohort uses supervised token training on self-sampled data,
not a separate policy-gradient objective. SFT tags can describe the same data
construction, and one self-sampled SFT recipe deliberately retains wrong attempts.
Keep objective, recorded tag and independent data-treatment declarations separate.

Only whitelisted first-plan data.source/data.selection phrases and explicit
data.build_command wrong-fraction arguments are inspected. No source code is
executed or searched, and no result, count, path, ID or arbitrary text is emitted.
True means a positive prospective declaration; None means unresolved, never false.
This is not a claim of execution, data purity, verified-only exposure or exact mix.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from collections.abc import Mapping

from tools.outcome_prediction.wm_code_benchmark import positive_input

SCHEMA = "matched-treatment-evidence-v1"
EVIDENCE_KEYS = ("self_sampled", "correctness_selection", "wrong_attempts", "gold_replay", "teacher_generated")
FAMILY_ALIASES = {
    "sft": "sft", "supervised": "sft", "rft": "rft",
    "distill": "distill", "distillation": "distill",
    "rl": "rl", "grpo": "grpo", "ppo": "ppo", "dpo": "dpo",
    "merge": "merge", "merging": "merge", "decoding": "decoding", "decode": "decoding",
}
FAMILY_TOKENS = tuple(sorted(set(FAMILY_ALIASES.values())))
SUPERVISED_TOKENS = frozenset({"sft", "rft", "distill"})
OBJECTIVES = ("supervised_token_training", "unknown")
STATUSES = ("available", "missing_recipe", "quarantined", "no_parent_recipe")
PATTERNS = {
    "self_sampled": (
        r"\bsynthetic\s*:\s*self\b",
        r"\bself[- ](?:generated|sampled|sampling)\b",
        r"\bon[- ]policy\b",
        r"\bmodel['’]s own (?:sampled )?(?:text|solutions?|samples?|completions?)\b",
    ),
    "correctness_selection": (
        r"\b(?:answer|gold)[- ]verified\b",
        r"\bcorrectness[- ]filter(?:ed|ing)?\b",
        (r"\b(?:keep|kept|retain(?:ed)?|filter(?:ed|ing)?|verif(?:y|ied))\b[^.;\n]{0,70}"
         r"\b(?:correct|matches? (?:the )?gold|equals? (?:the )?gold)\b"),
        (r"\b(?:only|shortest|best|on[- ]policy|rejection[- ]sampled)\s+(?:\w+\s+){0,3}correct\s+"
         r"(?:solutions?|completions?|samples?|derivations?|rft)\b"),
        r"\bcorrect[- ]slot\s+top[- ]ups?\b",
        r"\bcorrect first[- ]answer\b",
        r"\bkeep\s+(?:last[- ]number|last number)\s*==\s*gold\b",
    ),
    "wrong_attempts": (
        (r"\b(?:includ(?:e|es|ing)|retain(?:ed|s|ing)?|keep|kept|mix(?:ed|es|ing)?|fill(?:s|ed|ing)?)\b"
         r"[^.;\n]{0,85}\b(?:wrong|incorrect)\s+(?:on[- ]policy\s+)?(?:attempts?|solutions?|samples?|approaches?)\b"),
    ),
    "gold_replay": (
        r"\b(?:gold|reference)\b[^.\n]{0,60}\b(?:replay|anchor|fallback)\b",
        r"\b(?:replay|anchor|fallback)\b[^.\n]{0,60}\b(?:gold|reference)\b",
        r"(?:\bplus|\+)\s+(?:\d[\d,]*\s+)?gold\s+(?:gsm8k|training|train|reasoning|solutions?|examples?|data)\b",
        r"\bgold\s+(?:gsm8k\s+)?train(?:ing)?\s*\(fallback\)",
    ),
    "teacher_generated": (
        r"\bteacher[- ]generated\b",
        r"\bgenerated\s+(?:by|with)\s+(?:(?:a|the)\s+)?teacher\b",
        r"\bteacher(?:\s+model)?['’]?s?\s+(?:samples?|solutions?|completions?|reasoning)\b",
        r"\bdistilled\s+from\s+(?:(?:a|the)\s+)?teacher\b",
    ),
}
WRONG_FLAGS = frozenset({"--wrong-frac", "--wrong_frac", "--incorrect-fraction", "--incorrect_fraction"})


def _sha(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _positive_matches(text, patterns):
    """Conservative local negation/scope screen; unresolved cases stay unknown."""
    if not isinstance(text, str):
        return []
    text = text.casefold()
    matches = []
    for number, pattern in enumerate(patterns):
        for match in re.finditer(pattern, text):
            # Nearby negations and exclusion verbs conservatively suppress evidence.
            before = re.split(r"[.;\n]", text[max(0, match.start() - 100):match.start()])[-1]
            inside = match.group()
            after = text[match.end():match.end() + 65]
            if re.search(r"\b(?:no|not|never|without|disable\w*|exclude\w*|avoid\w*|drop\w*|discard\w*|remove\w*)\b"
                         r"(?:\W+\w+){0,8}\W*$", before):
                continue
            if re.search(r"\b(?:no|not|never|without|exclude\w*|drop\w*|discard\w*)\b", inside):
                continue
            if re.match(r"\s*(?:\w+\s+){0,3}(?:is|are|were|was)\s+(?:not\s+(?:used|kept|retained)|excluded|discarded|removed)", after):
                continue
            # References explicitly scoped to evaluation are not training sources.
            scope = before + inside
            if re.search(r"\b(?:evaluation|eval|test)[- ]only\b|\b(?:for|during)\s+(?:evaluation|testing|eval)\b", scope):
                continue
            matches.append({"pattern": number, "start": match.start(), "end": match.end()})
    return matches


def _wrong_fraction(command):
    """A declared builder argument, not an inferred default or executed setting."""
    if isinstance(command, str):
        try:
            argv = shlex.split(command)
        except ValueError:
            return False
    elif isinstance(command, (list, tuple)) and all(isinstance(x, str) for x in command):
        argv = list(command)
    else:
        return False
    if any(token in {";", "&&", "||", "|"} for token in argv):
        return False
    values = []
    for index, token in enumerate(argv):
        key, separator, value = token.partition("=")
        if key not in WRONG_FLAGS:
            continue
        if not separator:
            if index + 1 == len(argv):
                return False
            value = argv[index + 1]
        try:
            number = float(value)
        except ValueError:
            return False
        if not math.isfinite(number) or not 0 <= number <= 1:
            return False
        values.append(number)
    return bool(values) and len(set(values)) == 1 and values[0] > 0


def _empty(status):
    return {"status": status, "family_tokens": [], "objective": "unknown",
            "evidence": dict.fromkeys(EVIDENCE_KEYS)}


def _step(model_input, status):
    if status != "available" or not isinstance(model_input, Mapping):
        return _empty(status), {"status": status, "fields": []}
    # The established positive projection discards outcomes and unrelated prose;
    # we narrow it again to method.family and first-plan data declarations only.
    safe = positive_input({"model_input": model_input})
    setup = safe["plan"]["setup"]
    family = setup["method"].get("family")
    tokens = sorted({FAMILY_ALIASES[token] for token in re.findall(r"[a-z]+", family.casefold())
                     if token in FAMILY_ALIASES}) if isinstance(family, str) else []
    objective = "supervised_token_training" if tokens and set(tokens) <= SUPERVISED_TOKENS else "unknown"
    result = {**_empty("available"), "family_tokens": tokens, "objective": objective}
    provenance = {"status": "available", "family_sha256": _sha(family), "fields": []}
    for index, entry in enumerate(setup["data"]):
        for field in ("source", "selection"):
            value = entry.get(field)
            if not isinstance(value, str):
                continue
            positives = {name: matches for name, patterns in PATTERNS.items()
                         if (matches := _positive_matches(value, patterns))}
            provenance["fields"].append({"entry": index, "field": field, "sha256": _sha(value), "matches": positives})
            for name in positives:
                result["evidence"][name] = True
        if _wrong_fraction(entry.get("build_command")):
            result["evidence"]["wrong_attempts"] = True
            provenance["fields"].append({"entry": index, "field": "build_command", "sha256": _sha(entry["build_command"]),
                                          "matches": {"wrong_attempts": "explicit_positive_fraction_argument"}})
    return result, provenance


def extract(example):
    """Return prompt-safe context and separate, non-prompt hash/field provenance.

    Caller must supply the pinned, screened input bundle. Only the latest history
    entry can describe the immediate parent; missing/quarantined entries are never
    backfilled from older recipes or from the query's current method.
    """
    if not isinstance(example, Mapping):
        raise TypeError("Expected a screened input example")
    model_input = example.get("model_input")
    current, current_provenance = _step(model_input, "available" if isinstance(model_input, Mapping) else "missing_recipe")
    history = example.get("history", [])
    if not isinstance(history, (list, tuple)):
        raise TypeError("History must be a screened sequence")
    parent_input = None
    parent_status = "no_parent_recipe"
    if history:
        latest = history[-1]
        if not isinstance(latest, Mapping):
            parent_status = "missing_recipe"
        elif latest.get("recipe_status") != "screened":
            parent_status = "quarantined" if latest.get("recipe_status") == "quarantined" else "missing_recipe"
        elif isinstance(latest.get("model_input"), Mapping):
            parent_input, parent_status = latest["model_input"], "available"
        else:
            parent_status = "missing_recipe"
    parent, parent_provenance = _step(parent_input, parent_status)
    context = {"schema": SCHEMA, "current": current, "immediate_parent": parent}
    validate_context(context)
    return {"context": context,
            "provenance": {"schema": SCHEMA, "current": current_provenance, "immediate_parent": parent_provenance}}


def validate_context(context):
    """Reject any noncanonical or unapproved prompt metadata; return it unchanged."""
    if type(context) is not dict or set(context) != {"schema", "current", "immediate_parent"}:
        raise ValueError("Use exactly the fixed treatment context fields")
    if context["schema"] != SCHEMA:
        raise ValueError("Unknown treatment evidence schema")
    for scope in ("current", "immediate_parent"):
        step = context[scope]
        if type(step) is not dict or set(step) != {"status", "family_tokens", "objective", "evidence"}:
            raise ValueError("Use exactly the fixed treatment step fields")
        if type(step["status"]) is not str or step["status"] not in STATUSES:
            raise ValueError("Unknown recipe status")
        tokens = step["family_tokens"]
        if (type(tokens) is not list or any(type(token) is not str or token not in FAMILY_TOKENS for token in tokens)
                or tokens != sorted(set(tokens))):
            raise ValueError("Family tokens must be a canonical fixed-enum list")
        expected_objective = "supervised_token_training" if tokens and set(tokens) <= SUPERVISED_TOKENS else "unknown"
        if type(step["objective"]) is not str or step["objective"] != expected_objective:
            raise ValueError("Objective does not match the audited family grouping")
        flags = step["evidence"]
        if type(flags) is not dict or set(flags) != set(EVIDENCE_KEYS):
            raise ValueError("Use exactly the fixed positive evidence fields")
        if any(value is not True and value is not None for value in flags.values()):
            raise ValueError("Evidence must be True or None; no inferred negative values")
        if step["status"] != "available" and (tokens or any(value is not None for value in flags.values())):
            raise ValueError("Unavailable recipes cannot carry inferred treatment evidence")
    return context
