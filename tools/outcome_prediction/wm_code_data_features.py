"""Compact, outcome-blind data-recipe features from pre-proposal inputs.

The caller owns timestamp gating and supplies one ``model_input`` at a time.
Only ``plan.setup.data`` and reconstructed ``data_builder_*`` code are read.
Counts are *requested quotas/caps*, never realized corpus sizes: in particular
``n_examples``, result fields, free prose numbers, and generated yields are not
inputs. Policies below mean "positive evidence in the declared recipe", not
proof that a branch executed or a quality check succeeded.

No source code is executed. Argparse defaults require an identifiable invoked
builder, one parse_args call, and declarations outside conditional branches.
Known CLI overrides win. Different data entries with incompatible settings
produce unknown, not an average. Uppercase code constants are separately named
``code_declared`` and are not presented as resolved execution configuration.
"""

from __future__ import annotations

import ast
import math
import posixpath
import re
import shlex
from collections import defaultdict
from pathlib import PurePosixPath

# Normalize underscores only for explicitly listed long options. Generic --n
# stays distinct because it may count rows, questions, or generation requests.
OPTION_FIELDS = {
    "n-gsm": "quota.gsm",
    "n-aug-gsm": "quota.aug_gsm",
    "n-math": "quota.math",
    "n-aug-math": "quota.aug_math",
    "n-human": "quota.human",
    "n-omi": "quota.omi",
    "n-omr": "quota.omr",
    "n-or1": "quota.openr1",
    "openr1-n": "quota.openr1",
    "n-fresh": "quota.fresh",
    "n-replay": "quota.replay",
    "n": "quota.requested_n",
    "max-examples": "cap.examples",
    "limit": "cap.examples",
    "max-total": "cap.examples",
    "max-keep": "cap.examples",
    "max-per-problem": "cap.solutions_per_problem",
    "max-per-q": "cap.solutions_per_problem",
    "keep-per-problem": "cap.solutions_per_problem",
    "keep-per-q": "cap.solutions_per_problem",
    "rft-per-q": "cap.solutions_per_problem",
    "max-tokens": "cap.max_tokens",
    "max-tok": "cap.max_tokens",
    "max-completion-tokens": "cap.max_tokens",
    "max-assistant-tokens": "cap.max_tokens",
    "max-gen-tokens": "cap.max_tokens",
    "min-tokens": "cap.min_tokens",
    "min-tok": "cap.min_tokens",
    "max-chars": "cap.max_chars",
    "min-chars": "cap.min_chars",
    "max-total-tokens": "cap.total_tokens",
    "token-budget": "cap.total_tokens",
    "shard-start": "sampling.shard_start",
    "k": "generation.samples",
    "samples": "generation.samples",
    "temperature": "generation.temperature",
    "temp": "generation.temperature",
    "top-p": "generation.top_p",
    "fewshot-frac": "fewshot.fraction",
    "fewshot-prob": "fewshot.fraction",
    "fewshot": "fewshot.count",
    "nshot": "fewshot.count",
}
CONSTANT_FIELDS = {
    "MAXLEN": "code_declared.max_seq_len",
    "MAX_LEN": "code_declared.max_seq_len",
    "MAXTOK": "code_declared.max_tokens",
    "MAX_TOK": "code_declared.max_tokens",
    "MAX_GEN_TOK": "code_declared.max_tokens",
    "MINTOK": "code_declared.min_tokens",
    "MIN_RESP_TOK": "code_declared.min_tokens",
    "MIN_COMP_CHARS": "code_declared.min_chars",
    "MIN_CHARS": "code_declared.min_chars",
    "MAX_CHARS": "code_declared.max_chars",
    "MAX_CHAR": "code_declared.max_chars",
    "MAX_KEEP": "code_declared.max_keep",
}
SOURCE_PATTERNS = {
    "gsm8k": r"\bgsm8k\b",
    "math": r"(?<![a-z])(?:augmented[_ -])?math(?![a-z])",
    "openmathinstruct": r"openmathinstruct",
    "openmathreasoning": r"openmathreasoning",
    "openr1": r"open[-_ ]?r1",
    "metamath": r"meta[-_ ]?math",
    "synthetic": r"\b(?:synthetic|self[- ]sampled|self[- ]generated)\b",
    "derived": r"\bderived\s*:",
}
POLICY_PATTERNS = {
    "correctness_filter": (
        r"\b(?:filter(?:ed|ing)?|keep|kept|retain(?:ed)?|verif(?:y|ied))"
        r"[^.;\n]{0,70}\b(?:correct|matches? (?:the )?gold|equals? (?:the )?gold)\b"
        r"|\b(?:answer[- ]verified|correctness[- ]filter(?:ed|ing)?)\b"
    ),
    "deduplicate": r"\b(?:dedup(?:e|ed|ing|lication|licated)?|deduplicat\w*)\b",
    "exclude_prior": (
        r"\b(?:exclude[ds]?|disjoint|unseen|not seen|no (?:training )?example repeats)\b"
        r"[^.;\n]{0,65}\b(?:prior|previous|earlier|exp[-_]|stage|training|seen)"
        r"|\b(?:fresh|non[- ]overlapping)\s+(?:shards?|examples?|rows?|data)\b"
    ),
    "train_only": (
        r"\btrain[- ]only\b|\btrain\s+split\s+only\b"
        r"|\b(?:test|evaluation)\s+split\s+is\s+never\s+(?:read|touched|used)\b"
    ),
    "numeric_answer": r"\bnumeric\s+(?:expected[_ ]answer|answers?)\b",
    "answer_line": r"(?:append|trailing|end(?:ing)?|convert|rewrite|rewritten)[^.;\n]{0,70}answer\s*:",
    "boxed_required": r"\b(?:require[ds]?|contain[s]?|containing)\b[^.;\n]{0,35}\\?boxed",
    "strip_calculator": r"\b(?:strip\w*|remov\w*)[^.;\n]{0,40}(?:calculator|<<)",
}
POLICY_NUMERIC_PATTERNS = {
    "fewshot.fraction": (
        (
            r"(?P<percent>\d+(?:\.\d+)?)%\s+of\s+(?:rows|examples|prompts)\s+"
            r"(?:get|receive|have|are given)\s+[^.;\n]{0,35}\b(?:few[- ]shot|k[- ]shot)\b"
        ),
        (
            r"\bwith\s+p\s*=\s*(?P<fraction>0?\.\d+|1(?:\.0+)?)\s+"
            r"(?:a|an)\s+(?:few[- ]shot|k[- ]shot)\s+prefix\b"
        ),
    ),
    "cap.solutions_per_problem": (
        (
            r"\bat\s+most\s+(?P<count>\d+|one)\s+(?:distinct\s+)?solutions?\s+per\s+"
            r"(?:distinct\s+|unique\s+)?(?:problem|question)\b"
        ),
    ),
}
NUMERIC_FIELDS = tuple(sorted(set(OPTION_FIELDS.values()) | set(CONSTANT_FIELDS.values())))
DERIVED_FIELDS = (
    "mixture.weight_count",
    "mixture.weight_entropy",
    "mixture.max_weight",
    "mixture.quota_source_count",
    "mixture.quota_sum",
    "mixture.quota_entropy",
    "mixture.math_fraction",
    "mixture.augmented_fraction",
    "mixture.fresh_fraction",
)


def _number(value, field):
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and not re.fullmatch(
        r"[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value.strip()
    ):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or not 0 <= value <= 1e12:
        return None
    if field in ("fewshot.fraction", "generation.top_p") and value > 1:
        return None
    if field == "generation.temperature" and value > 10:
        return None
    return value


def _argv(command):
    if isinstance(command, str):
        try:
            command = shlex.split(command)
        except ValueError:
            return None
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        return None
    # No command composition, command substitutions, python -c, or shell payloads.
    if any(
        x in (";", "&&", "||", "|", "-c", "-lc") or "$(" in x or "`" in x or "\n" in x or ";" in x
        for x in command
    ):
        return None
    return command


def _cli_values(argv):
    found, explicitly_set = defaultdict(list), set()
    for i, token in enumerate(argv or []):
        if not token.startswith("--"):
            continue
        name, sep, raw = token[2:].partition("=")
        field = OPTION_FIELDS.get(name.replace("_", "-"))
        if field is None:
            continue
        explicitly_set.add(field)
        if not sep:
            raw = argv[i + 1] if i + 1 < len(argv) else None
        value = _number(raw, field)
        found[field].append(value)
    return found, explicitly_set


def _is_main_guard(node):
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and ast.dump(node.test, include_attributes=False)
        in {
            ast.dump(ast.parse("__name__ == '__main__'", mode="eval").body),
            ast.dump(ast.parse("'__main__' == __name__", mode="eval").body),
        }
    )


def _code_settings(script, invoked):
    """Return declarations; dynamic control flow never becomes known config."""
    audit = {"parsed": False, "defaults_allowed": False, "conditional_defaults_skipped": 0}
    defaults, constants = defaultdict(list), defaultdict(list)
    if not invoked or script.get("status") != "reconstructed":
        return defaults, constants, audit
    try:
        tree = ast.parse(script.get("content") or "")
    except (SyntaxError, ValueError, TypeError):
        return defaults, constants, audit
    audit["parsed"] = True
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def unconditional(node):
        ancestor = parents.get(node)
        while ancestor is not None:
            if isinstance(
                ancestor,
                (
                    ast.If,
                    ast.For,
                    ast.While,
                    ast.Try,
                    ast.TryStar,
                    ast.With,
                    ast.IfExp,
                    ast.Match,
                    ast.ClassDef,
                    ast.Lambda,
                    ast.ListComp,
                    ast.SetComp,
                    ast.DictComp,
                    ast.GeneratorExp,
                ),
            ) and not _is_main_guard(ancestor):
                return False
            if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                calls = [
                    statement.value
                    for top in tree.body
                    if _is_main_guard(top)
                    for statement in top.body
                    if isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Name)
                    and statement.value.func.id == ancestor.name
                ]
                if ancestor.name != "main" or len(calls) != 1:
                    return False
            ancestor = parents.get(ancestor)
        return True

    assignments = defaultdict(list)
    loaded_names = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in CONSTANT_FIELDS:
                    value = _number(node.value.value, CONSTANT_FIELDS[target.id])
                    if value is not None and target.id in loaded_names:
                        assignments[CONSTANT_FIELDS[target.id]].append(value)
    constants.update(assignments)
    parse_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "parse_args"
    ]
    # Explicit parse_args([...]) may replace actual argv. Subparsers introduce
    # branch-dependent defaults even without a Python if statement.
    if (
        len(parse_calls) != 1
        or parse_calls[0].args
        or parse_calls[0].keywords
        or not unconditional(parse_calls[0])
        or any(
            isinstance(n, ast.Attribute) and n.attr in ("add_subparsers", "set_defaults")
            for n in ast.walk(tree)
        )
    ):
        return defaults, constants, audit
    audit["defaults_allowed"] = True
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        if ast.dump(node.func.value) != ast.dump(parse_calls[0].func.value):
            continue
        if not unconditional(node):
            audit["conditional_defaults_skipped"] += 1
            continue
        default = next((k.value for k in node.keywords if k.arg == "default"), None)
        if not isinstance(default, ast.Constant):
            continue
        fields = {
            OPTION_FIELDS.get(arg.value[2:].replace("_", "-"))
            for arg in node.args
            if isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("--")
        }
        for field in fields - {None}:
            value = _number(default.value, field)
            if value is not None:
                defaults[field].append(value)
    return defaults, constants, audit


def _entropy(values):
    total = sum(values)
    return -sum((v / total) * math.log(v / total) for v in values if v > 0) if total > 0 else None


def _invokes(argv, path, cwd):
    if not argv or not path:
        return False
    first = PurePosixPath(argv[0]).name
    if re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", first):
        candidate = argv[1] if len(argv) > 1 else ""
    elif argv[0].endswith(".py"):
        candidate = argv[0]
    else:
        return False

    def resolve(value):
        if not value or "$" in value or "~" in value:
            return None
        if value.startswith("/"):
            return posixpath.normpath(value)
        if isinstance(cwd, str) and cwd.startswith("/") and "$" not in cwd and "~" not in cwd:
            return posixpath.normpath(posixpath.join(cwd, value))
        # No inferred working directory and no basename-only routing.
        return None

    expected, actual = resolve(path), resolve(candidate)
    return expected is not None and actual is not None and expected == actual


def _positive_match(pattern, text):
    """Do not turn explicit local negation into positive policy evidence."""
    for match in re.finditer(pattern, text):
        if _not_locally_negated(text, match.start()):
            return True
    return False


def _not_locally_negated(text, start):
    preceding = text[max(0, start - 35) : start]
    return not re.search(r"\b(?:no|not|without|never|disable[ds]?)\s+(?:\w+\s+){0,2}$", preceding)


def extract_data_features(model_input: dict) -> tuple[dict[str, float | None], dict]:
    """Extract a fixed schema; audit contains provenance, not predictive text."""
    plan = model_input.get("plan") or {}
    setup = plan.get("setup") or {}
    cwd = (setup.get("command") or {}).get("cwd")
    family = str((setup.get("method") or {}).get("family") or "").lower()
    training = bool(re.search(r"\b(?:sft|rft|grpo|ppo|dpo|rl|distill)(?:\b|_)", family))
    nontraining = not training and bool(
        re.search(
            r"decode|decoding|merge|model_soup|weight_averaging|checkpoint_selection|eval", family
        )
    )
    data = [d for d in setup.get("data") or [] if isinstance(d, dict)]
    scripts = [
        c
        for c in model_input.get("code") or []
        if isinstance(c, dict) and str(c.get("role", "")).startswith("data_builder_")
    ]
    out = {key: None for key in (*NUMERIC_FIELDS, *DERIVED_FIELDS)}
    out.update({"source.evidence." + key: 0.0 for key in SOURCE_PATTERNS})
    out.update({"policy.evidence." + key: 0.0 for key in POLICY_PATTERNS})
    audit = {
        "version": 1,
        "entries": [],
        "field_sources": {},
        "conflicts": [],
        "ignored_fields": ["n_examples", "result", "progress", "paths", "free_text_counts"],
        "numeric_semantics": "planned quotas/caps, not actual generated yields",
        "data_intervention_active": False if nontraining else (True if training else None),
        "routing": "normalized exact paths using declared cwd; no inferred cwd or basename fallback",
    }
    collected = defaultdict(list)
    weights = []
    command_count = builder_count = default_count = 0
    for index, entry in enumerate(data):
        argv = None if nontraining else _argv(entry.get("build_command"))
        values, explicitly_set = _cli_values(argv)
        resolved_here = set(explicitly_set)
        command_count += bool(argv)
        record = {
            "entry": index,
            "command_parseable": argv is not None,
            "builders": [],
            "inactive_data_recipe": nontraining,
        }
        for key, items in values.items():
            collected[key].extend(items)
            audit["field_sources"].setdefault(key, []).append({"entry": index, "source": "argv"})
        matching = [c for c in scripts if c.get("role") == f"data_builder_{index}"]
        for script in matching:
            path = str(script.get("script_path") or "")
            # Exact path or basename invocation; paths are only routing metadata.
            invoked = _invokes(argv, path, cwd)
            defaults, constants, code_audit = _code_settings(script, invoked)
            record["builders"].append(code_audit)
            builder_count += code_audit["parsed"]
            for key, items in defaults.items():
                if key not in explicitly_set:
                    resolved_here.add(key)
                    collected[key].extend(items)
                    default_count += 1
                    audit["field_sources"].setdefault(key, []).append(
                        {"entry": index, "source": "argparse_default"}
                    )
            for key, items in constants.items():
                collected[key].extend(items)
                audit["field_sources"].setdefault(key, []).append(
                    {"entry": index, "source": "referenced_module_constant"}
                )
        weight = None if nontraining else _number(entry.get("mixture_weight"), "mixture_weight")
        if weight is not None:
            weights.append(weight)
        source = str(entry.get("source") or "").lower()
        for key, pattern in SOURCE_PATTERNS.items():
            if _positive_match(pattern, source):
                out["source.evidence." + key] = 1.0
        # Numeric prose counts deliberately excluded. These fixed policy markers
        # may be present without proving actual branch execution or success.
        selection = "" if nontraining else str(entry.get("selection") or "").lower()
        for field, patterns in POLICY_NUMERIC_PATTERNS.items():
            if field in resolved_here:
                continue
            for pattern in patterns:
                for match in re.finditer(pattern, selection):
                    if not _not_locally_negated(selection, match.start()):
                        continue
                    groups = match.groupdict()
                    raw = next(value for value in groups.values() if value is not None)
                    value = _number(1 if raw == "one" else raw, field)
                    # Percent is validated as a fraction only after division.
                    if groups.get("percent") is not None:
                        value = _number(float(raw) / 100, field)
                    if value is not None:
                        collected[field].append(value)
                        audit["field_sources"].setdefault(field, []).append(
                            {"entry": index, "source": "bounded_policy_expression"}
                        )
        for key, pattern in POLICY_PATTERNS.items():
            if _positive_match(pattern, selection):
                out["policy.evidence." + key] = 1.0
        audit["entries"].append(record)
    for key, values in collected.items():
        unique = set(values)
        if len(unique) == 1:
            out[key] = values[0]
        else:
            audit["conflicts"].append(key)
    if weights:
        out["mixture.weight_count"] = float(len(weights))
        out["mixture.weight_entropy"] = _entropy(weights)
        out["mixture.max_weight"] = max(weights) / sum(weights) if sum(weights) else None
    # Restrict disjoint quota summaries to explicitly named source buckets. OMI
    # and GSM may overlap, so they are never added into a bogus corpus total.
    buckets = ["gsm", "aug_gsm", "math", "aug_math"]
    q = {key: out["quota." + key] for key in buckets}
    known = [value for value in q.values() if value is not None]
    if known:
        out["mixture.quota_source_count"] = float(len(known))
        out["mixture.quota_sum"] = sum(known)
        out["mixture.quota_entropy"] = _entropy(known)
    # Fractions need all four requested quotas; omitted flags are not assumed 0.
    if len(known) == len(buckets) and sum(known) > 0:
        out["mixture.math_fraction"] = (q["math"] + q["aug_math"]) / sum(known)
        out["mixture.augmented_fraction"] = (q["aug_gsm"] + q["aug_math"]) / sum(known)
    fresh, replay = out["quota.fresh"], out["quota.replay"]
    if fresh is not None and replay is not None and fresh + replay > 0:
        out["mixture.fresh_fraction"] = fresh / (fresh + replay)
    out.update(
        {
            "availability.data_entries": float(len(data)),
            "availability.build_commands": float(command_count),
            "availability.invoked_builders": float(builder_count),
            "availability.default_fields": float(default_count),
            "availability.conflicting_fields": float(len(audit["conflicts"])),
        }
    )
    return out, audit
