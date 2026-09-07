"""Fixed training-dataset family evidence from screened prospective recipes.

The caller verifies proposal timestamps and code reconstruction provenance. This
extractor reads only plan.setup.data.source and reconstructed data_builder_N code;
it never infers training data from the evaluation benchmark, reads scores, executes
code, or forwards arbitrary source text. Identity means declaration/reference
evidence, NOT proof of executed training, complete component coverage or mixture
proportions. Unknown current recipes are never backfilled from ancestors.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping

FAMILIES = (
    "gsm8k", "metamathqa", "openmathinstruct2", "openmathreasoning", "openr1_math220k",
    "mixture_of_thoughts", "numinamath_cot", "s1k", "aimo_validation_aime",
    "openthoughts3", "hendrycks_math", "orca_math",
)
DATASETS = {
    "gsm8k": "openai/gsm8k",
    "metamathqa": "meta-math/MetaMathQA",
    "openmathinstruct2": "nvidia/OpenMathInstruct-2",
    "openmathreasoning": "nvidia/OpenMathReasoning",
    "openr1_math220k": "open-r1/OpenR1-Math-220k",
    "mixture_of_thoughts": "open-r1/Mixture-of-Thoughts",
    "numinamath_cot": "AI-MO/NuminaMath-CoT",
    "s1k": "simplescaling/s1K-1.1",
    "aimo_validation_aime": "AI-MO/aimo-validation-aime",
    "openthoughts3": "open-thoughts/OpenThoughts3-1.2M",
    "hendrycks_math": "EleutherAI/hendrycks_math",
    "orca_math": "microsoft/orca-math-word-problems-200k",
}
FLAGS = (
    "context_available", "identity_known", "identity_unknown", "multiple_families",
    "mixture_declared", "synthetic_declared", "derived_declared", "code_only_identity",
)
FEATURE_KEYS = tuple(sorted((
    *(f"recipe_data.{scope}.family.{family}" for scope in ("current", "parent") for family in FAMILIES),
    *(f"recipe_data.{scope}.{flag}" for scope in ("current", "parent") for flag in FLAGS),
    *(f"recipe_data.history.any_family.{family}" for family in FAMILIES),
    *("recipe_data.history." + key for key in ("steps", "screened_steps", "known_identity_steps", "unresolved_steps", "complete_to_base")),
)))
_REPO_TO_FAMILY = {repo.casefold(): family for family, repo in DATASETS.items()}
_ALIASES = {
    "gsm8k": (r"openai/gsm8k", r"gsm8k[ _-]+train(?:ing)?", r"gold[ _-]+gsm8k"),
    "metamathqa": (r"(?:meta-math/)?metamathqa",),
    "openmathinstruct2": (r"(?:nvidia/)?openmathinstruct-2",),
    "openmathreasoning": (r"(?:nvidia/)?openmathreasoning",),
    "openr1_math220k": (r"(?:open-r1/)?openr1-math-220k",),
    "mixture_of_thoughts": (r"(?:open-r1/)?mixture-of-thoughts",),
    "numinamath_cot": (r"(?:ai-mo/)?numinamath-cot",),
    "s1k": (r"(?:simplescaling/)?s1k(?:-1\.1)?",),
    "aimo_validation_aime": (r"(?:ai-mo/)?aimo-validation-aime",),
    "openthoughts3": (r"(?:open-thoughts/)?openthoughts3-1\.2m",),
    "hendrycks_math": (r"(?:eleutherai/)?hendrycks_math",),
    "orca_math": (r"(?:microsoft/)?orca-math-word-problems-200k",),
}
_PLAN_PATTERNS = {
    family: re.compile(r"(?<![\w/])(?:" + "|".join(patterns) + r")(?![\w/])", re.IGNORECASE)
    for family, patterns in _ALIASES.items()
}
_EVAL_NAME = re.compile(r"(?:^|_)(?:eval(?:uate|uation)?|test|dev|benchmark)(?:_|$)", re.IGNORECASE)
_NONTRAIN_SPLIT = re.compile(r"(?:^|/)(?:test|validation|valid|dev|eval)(?:$|[\[_.\-/])", re.IGNORECASE)


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plan_families(value):
    return {family for family, pattern in _PLAN_PATTERNS.items() if pattern.search(value)}


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return prefix + "." + node.attr if prefix else None
    return None


def _bindings(tree):
    """Only unambiguous module-level string constants; no scope inference."""
    assigned = defaultdict(list)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned[target.id].append(node.value)
    stores = defaultdict(int)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            stores[node.id] += 1
        elif isinstance(node, ast.arg):
            stores[node.arg] += 1
    return {name: nodes[0].value for name, nodes in assigned.items()
            if len(nodes) == 1 and stores[name] == 1 and isinstance(nodes[0], ast.Constant) and isinstance(nodes[0].value, str)}


def _literal(node, bindings, depth=0):
    if depth > 8:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal(node.left, bindings, depth + 1), _literal(node.right, bindings, depth + 1)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.Call) and _name(node.func) in {"os.path.expanduser", "Path", "pathlib.Path"} and len(node.args) == 1:
        return _literal(node.args[0], bindings, depth + 1)
    return None


def _imports(tree):
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                names[alias.asname or alias.name] = node.module + "." + alias.name
    shadowed = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    shadowed.update(node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    shadowed.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
    return {name: value for name, value in names.items() if name not in shadowed}


def _qualified(node, imports):
    name = _name(node)
    if not name:
        return None
    first, dot, rest = name.partition(".")
    return imports[first] + (dot + rest if dot else "") if first in imports else None


def _excluded_context(node, parents):
    child = node
    ancestor = parents.get(child)
    while ancestor is not None:
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)) and _EVAL_NAME.search(ancestor.name):
            return True
        if isinstance(ancestor, ast.Assign) and any(isinstance(target, ast.Name) and _EVAL_NAME.search(target.id) for target in ancestor.targets):
            return True
        if isinstance(ancestor, ast.Call) and _EVAL_NAME.search((_name(ancestor.func) or "").split(".")[-1]):
            return True
        if isinstance(ancestor, ast.If) and isinstance(ancestor.test, ast.Constant):
            if not bool(ancestor.test.value) and child in ancestor.body:
                return True
            if bool(ancestor.test.value) and child in ancestor.orelse:
                return True
        child, ancestor = ancestor, parents.get(ancestor)
    return False


def _cache_family(value):
    match = re.search(r"(?:^|/)datasets--([^/]+)--([^/]+)/snapshots/", value)
    if not match:
        return None
    return _REPO_TO_FAMILY.get((match[1] + "/" + match[2]).casefold())


def _builder_references(code):
    text = code.get("content")
    if not isinstance(text, str):
        return [], {"parsed": False, "reason": "missing_code"}
    audit = {"code_sha256": _sha(text), "parsed": False, "excluded_nontraining_calls": 0,
             "unmapped_dataset_calls": 0}
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        audit["reason"] = "parse_failed"
        return [], audit
    audit["parsed"] = True
    bindings, imports = _bindings(tree), _imports(tree)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    references = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _excluded_context(node, parents):
            continue
        qualified = _qualified(node.func, imports)
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
        family, source_kind = None, None
        if qualified in {"datasets.load_dataset", "huggingface_hub.snapshot_download", "huggingface_hub.hf_hub_download"}:
            key = "path" if qualified == "datasets.load_dataset" else "repo_id"
            identifier = _literal(keywords.get(key, node.args[0] if node.args else None), bindings)
            split = _literal(keywords.get("split"), bindings)
            container = parents.get(node)
            if isinstance(container, ast.Subscript) and container.value is node:
                split = _literal(container.slice, bindings) or split
            if split and _NONTRAIN_SPLIT.search(split):
                audit["excluded_nontraining_calls"] += 1
                continue
            if qualified != "datasets.load_dataset" and _literal(keywords.get("repo_type"), bindings) != "dataset":
                continue
            if identifier is not None:
                family = _REPO_TO_FAMILY.get(identifier.casefold())
                if family is None:
                    audit["unmapped_dataset_calls"] += 1
            source_kind = "builder_dataset_loader_reference"
        elif qualified in {"glob.glob", "glob.iglob", "pandas.read_parquet", "pyarrow.parquet.read_table", "pyarrow.parquet.ParquetFile"}:
            value = _literal(node.args[0], bindings) if node.args else None
            if value is not None and not _NONTRAIN_SPLIT.search(value):
                family = _cache_family(value)
            source_kind = "builder_dataset_cache_reference"
        if family is not None:
            references.append({"family": family, "source": source_kind, "line": node.lineno})
    return references, audit


def _step(model_input, *, status):
    available = status == "screened" and isinstance(model_input, Mapping)
    source = model_input if available else {}
    plan = source.get("plan") or {}
    setup = plan.get("setup") or {}
    data = setup.get("data") or []
    if not isinstance(data, list):
        raise TypeError("Screened recipe data must be a list")
    evidence, provenance, plan_families, code_families = [], [], set(), set()
    mixture, synthetic, derived, unmapped = False, False, False, False
    for index, entry in enumerate(data):
        if not isinstance(entry, Mapping):
            continue
        text = entry.get("source")
        if not isinstance(text, str) or not text.strip():
            unmapped = True
            continue
        found = _plan_families(text)
        plan_families.update(found)
        unmapped |= not bool(found)
        # These are only declaration flags, never measured mixing ratios.
        mixture |= bool(re.search(r"\bmix(?:ture)?\s*:|\s\+\s", text, re.IGNORECASE))
        synthetic |= bool(re.search(r"\bsynthetic\s*:|\bself[- ](?:generated|sampled)\b|\bself\s*\(", text, re.IGNORECASE))
        derived |= bool(re.search(r"\bderived\b|\breplay\b", text, re.IGNORECASE))
        for family in sorted(found):
            evidence.append({"family": family, "source": "plan_data_source"})
        provenance.append({"kind": "plan_data_source", "entry_index": index,
                           "source_sha256": _sha(text), "families": sorted(found)})
    codes = source.get("code") or []
    if not isinstance(codes, list):
        raise TypeError("Screened recipe code must be a list")
    for index, code in enumerate(codes):
        if (not isinstance(code, Mapping) or code.get("status") != "reconstructed"
                or not re.fullmatch(r"data_builder_\d+", str(code.get("role", "")))):
            continue
        references, audit = _builder_references(code)
        code_families.update(item["family"] for item in references)
        evidence.extend({"family": item["family"], "source": item["source"]} for item in references)
        provenance.append({"kind": "data_builder", "code_index": index, **audit, "references": references})
    families = plan_families | code_families
    flags = {
        "context_available": float(available), "identity_known": float(bool(families)),
        "identity_unknown": float(not families), "multiple_families": float(len(families) > 1),
        "mixture_declared": float(mixture), "synthetic_declared": float(synthetic),
        "derived_declared": float(derived), "code_only_identity": float(bool(code_families) and not plan_families),
    }
    evidence = [{"family": family, "source": kind} for family, kind in sorted({(e["family"], e["source"]) for e in evidence})]
    context = {
        "status": "identified_reference" if families else ("unknown_identity" if available else ("missing_recipe" if status == "screened" else status)),
        "families": sorted(families), "canonical_family_representatives": [DATASETS[k] for k in sorted(families)],
        "evidence": evidence, "multiple_family_references": len(families) > 1,
        "mixture_declaration": mixture, "synthetic_declaration": synthetic,
        "derived_or_replay_declaration": derived, "unmapped_declared_source_present": unmapped,
        "quantities_and_mixture_ratios": "not_extracted",
        "dataset_versions_and_revisions": "not_resolved_family_representatives_only",
        "meaning": "declared_or_builder_referenced_families_not_verified_execution_or_complete_mixture",
    }
    return families, flags, context, provenance


def extract(example):
    """Return 57 finite scalars, bounded readable context and separate provenance."""
    if not isinstance(example, Mapping):
        raise TypeError("A screened example mapping is required")
    current = _step(example.get("model_input"), status="screened")
    history = example.get("history", [])
    if not isinstance(history, list):
        raise TypeError("History must be an oldest-to-newest list")
    history_steps = []
    for entry in history:
        if not isinstance(entry, Mapping) or entry.get("recipe_status") not in {"screened", "quarantined"}:
            raise ValueError("History recipes require screened/quarantined status")
        history_steps.append(_step(entry.get("model_input"), status=entry["recipe_status"]))
    parent = history_steps[-1] if history_steps else _step(None, status="no_parent_recipe")
    complete = example.get("history_complete_to_base", False)
    if type(complete) is not bool:
        raise TypeError("History completeness must be boolean")
    features = {}
    for scope, (families, flags, _, _) in (("current", current), ("parent", parent)):
        features.update({f"recipe_data.{scope}.family.{family}": float(family in families) for family in FAMILIES})
        features.update({f"recipe_data.{scope}.{key}": value for key, value in flags.items()})
    history_union = set().union(*(step[0] for step in history_steps))
    features.update({f"recipe_data.history.any_family.{family}": float(family in history_union) for family in FAMILIES})
    features.update({
        "recipe_data.history.steps": float(len(history_steps)),
        "recipe_data.history.screened_steps": float(sum(step[1]["context_available"] for step in history_steps)),
        "recipe_data.history.known_identity_steps": float(sum(bool(step[0]) for step in history_steps)),
        "recipe_data.history.unresolved_steps": float(sum(not step[0] for step in history_steps)),
        "recipe_data.history.complete_to_base": float(complete),
    })
    if tuple(sorted(features)) != FEATURE_KEYS or any(type(v) is not float or not math.isfinite(v) for v in features.values()):
        raise ValueError("Invalid fixed numeric dataset schema")
    return {
        "features": {key: features[key] for key in FEATURE_KEYS},
        "context": {
            "current": current[2], "immediate_parent_recipe": parent[2],
            "history_family_union": sorted(history_union),
            "history_recipe_count": len(history_steps), "history_complete_to_base": complete,
            "history_has_unresolved_dataset_identity": any(not step[0] for step in history_steps),
            "warning": "Training_dataset_references_only_not_evaluation_benchmark_not_actual_sizes_or_ratios",
        },
        "provenance": {"version": 1, "current": current[3],
                       "history": [step[3] for step in history_steps],
                       "context_sha256": _sha(json.dumps({"current": current[2], "parent": parent[2], "history_union": sorted(history_union)}, sort_keys=True))},
    }
