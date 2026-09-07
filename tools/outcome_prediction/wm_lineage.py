"""Label-value-blind, DECLARED artifact lineage; not verified execution lineage.

``build_graph(rows)`` reads identities, proposal times, task/base model, and a
positive projection of the first setup's path/command/dependency fields. It never
reads labels, eligibility, result/history fields, plan prose, recovered code, or
final artifacts. Data-source/origin prose contributes only explicit card IDs.

Command paths take precedence over parent-origin hints. Exact input artifact
variants are retained even when several merge ingredients share one producer.
Time-qualified declared output aliases are provisional: a null checkpoint hash
does not reject a declaration, but no edge is represented as content-verified.
Missing/dynamic/ambiguous paths stay unresolved, never silently become the base.
GSM MERGE_PARENTS is a crosscheck only, not a source of invented ingredients.

    python -m tools.outcome_prediction.wm_lineage \
      --inventory data/analysis/wm_rpm/data_v3/private/inventory.jsonl \
      --output-dir data/analysis/wm_rpm/lineage_v1

The immutable artifact pins original inventory bytes separately from the blind
graph. Terminal flags are descriptive, not eligibility/target-selection rules.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import posixpath
import re
import shlex
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from tools.outcome_prediction.build_examples import MERGE_PARENTS

WEIGHT_FLAGS = {
    "model",
    "model_path",
    "model_name_or_path",
    "init",
    "init_from",
    "init_model",
    "base",
    "base_model",
    "checkpoint",
    "ckpt",
    "resume_from_checkpoint",
}
MERGE_FLAGS = {"models", "srcs", "inputs", "a", "b", "src", "ckpt"}
OUTPUT_FLAGS = {"out", "output", "output_dir", "dst", "destination", "save_dir"}
DATA_FLAGS = {"data", "data_path", "train_data", "dataset", "input", "input_file", "problems"}
CONFIG_FLAGS = {
    "config",
    "config_path",
    "config_file",
    "generation_config",
    "generation_config_path",
    "tokenizer_from",
    "tokenizer_path",
    "tokenizer_name_or_path",
}
CONFIG_NAMES = {
    "generation_config.json",
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
}
SEPARATORS = {";", "&&", "||", "|", "&"}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _time(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else None
    except (AttributeError, ValueError, TypeError):
        return None


def card_refs(value):
    """Only identifiers, including shorthand exp01/02/03; never return prose."""
    if not isinstance(value, str):
        return []
    found = set()
    for match in re.finditer(r"(?i)(?<![A-Za-z0-9])exp[-_ ]?0*(\d+)((?:/0*\d+\b)*)", value):
        found.add(f"exp-{int(match[1]):02d}")
        found.update(f"exp-{int(n):02d}" for n in re.findall(r"/(\d+)", match[2]))
    return sorted(found)


def _artifact(value, cwd):
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip().strip("\"'")
    if any(c in value for c in "$`{}*?\n\r") or value.startswith("~"):
        return value  # Retain the unresolved expression, never expand it.
    if value.startswith("/"):
        return posixpath.normpath(value)
    return posixpath.normpath(posixpath.join(cwd, value)) if cwd else posixpath.normpath(value)


def _dynamic(value):
    return not value or any(c in value for c in "$`{}*?\n\r") or value.startswith("~")


def _base_reference(value, base_model):
    """Recognize the declared repository or its exact HF snapshot spelling.

    This is declared identity, not model-byte integrity. Preserve the revision;
    arbitrary local directories must not silently become the base model.
    """
    if not isinstance(value, str) or not isinstance(base_model, str):
        return None
    if value == base_model:
        return {"base_reference_form": "repository_id", "base_revision": None}
    if len(base_model.split("/")) != 2:
        return None
    cache_name = "models--" + base_model.replace("/", "--")
    match = re.search(r"(?:^|/)" + re.escape(cache_name) + r"/snapshots/([0-9a-f]{40})/?$", value)
    if match:
        return {"base_reference_form": "declared_hf_cache_snapshot", "base_revision": match[1]}
    return None


def _segments(argv):
    """Tokenize only. Shell control flow/substitutions are never evaluated."""
    warnings = []
    if isinstance(argv, str):
        try:
            lexer = shlex.shlex(argv, posix=True, punctuation_chars=";&|")
            lexer.whitespace_split = True
            argv = list(lexer)
        except ValueError:
            return [], ["unparseable_command"]
    if not isinstance(argv, list) or not all(isinstance(t, str) for t in argv):
        return [], ["missing_command_argv"]
    if (
        argv
        and posixpath.basename(argv[0]) in {"bash", "sh", "zsh"}
        and any(t in {"-c", "-lc"} for t in argv[1:])
    ):
        position = next(i for i, t in enumerate(argv) if t in {"-c", "-lc"})
        if position + 1 >= len(argv):
            return [], ["missing_shell_command"]
        if "<<" in argv[position + 1]:
            return [], ["inline_shell_heredoc_not_resolved"]
        parts, warnings = _segments(argv[position + 1])
        if re.search(r"\b(?:for|while|if|case)\b", argv[position + 1]):
            warnings.append("shell_control_flow_not_resolved")
        return parts, warnings
    result, current = [], []
    for token in argv:
        if token in SEPARATORS:
            if current:
                result.append(current)
            current = []
        else:
            current.append(token)
    if current:
        result.append(current)
    return result, warnings


def _flags(tokens):
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue
        flag, equals, value = token[2:].partition("=")
        values = [value] if equals else []
        i += 1
        if not equals:
            while i < len(tokens) and not tokens[i].startswith("-"):
                values.append(tokens[i])
                i += 1
        result.append((flag.replace("-", "_"), values))
    return result


def _values(values):
    for value in values:
        for item in value.split(","):
            if item.strip():
                yield item.strip()


def _is_eval(tokens):
    return any(
        re.search(
            r"(?:eval|evaluate|sweep|select)[A-Za-z0-9_-]*\.(?:py|sh)$", posixpath.basename(t)
        )
        for t in tokens[:4]
    )


def _is_merge(tokens, family=""):
    return bool(re.search(r"merge|soup|averag", family, re.IGNORECASE)) or any(
        re.search(r"merge|soup|averag", posixpath.basename(t), re.IGNORECASE) for t in tokens[:4]
    )


def _project(row):
    """Positive field projection. No broad recursive walk of source data."""
    setup = _mapping(_mapping(_mapping(row.get("model_input")).get("plan")).get("setup"))
    command = _mapping(setup.get("command")) or _mapping(setup.get("launch"))
    command_field = "setup.command" if _mapping(setup.get("command")) else "setup.launch"
    cwd = command.get("cwd") if isinstance(command.get("cwd"), str) else None
    warnings = []
    if cwd is None and isinstance(command.get("script"), str) and command["script"].startswith("/"):
        script = command["script"]
        cwd = (
            script.split("/scripts/", 1)[0] if "/scripts/" in script else posixpath.dirname(script)
        )
        warnings.append("cwd_inferred_from_declared_script")
    task = _mapping(_mapping(row.get("model_input")).get("task"))
    base_model = setup.get("base_model") or task.get("base_model")
    base_model = base_model if isinstance(base_model, str) else None
    parent = _mapping(setup.get("parent_checkpoint"))
    family = _mapping(setup.get("method")).get("family")
    family = family if isinstance(family, str) else ""
    node = {
        "example_id": row["example_id"],
        "cell_id": row["cell_id"],
        "card_id": row["card_id"],
        "first_submitted_at": row.get("first_submitted_at"),
        "base_model": base_model,
        "cwd": cwd,
        "outputs": [],
        "references": [],
        "warnings": warnings,
        "origin_refs": card_refs(parent.get("origin")),
        "origin_declares_base": parent.get("origin") == "base_model",
        "artifact_hash_present": bool(parent.get("hash")),
    }

    def output(raw, kind, evidence):
        path = _artifact(raw, cwd)
        if path:
            value = {"artifact": path, "kind": kind, "evidence": evidence}
            if value not in node["outputs"]:
                node["outputs"].append(value)
        return path

    def reference(raw, kind, evidence, *, internal=False, card_only=False, coefficient=None):
        if raw is None:
            return
        value = {
            "reference": raw,
            "kind": kind,
            "evidence": evidence,
            "artifact": None if card_only or raw == base_model else _artifact(raw, cwd),
            "internal": internal,
            "card_only": card_only,
        }
        if coefficient is not None:
            value["merge_coefficient"] = coefficient
        node["references"].append(value)

    declared_out = setup.get("output_dir")
    if isinstance(declared_out, str):
        path = _artifact(declared_out, cwd)
        kind = (
            "evaluation_output"
            if re.search(r"/(?:logs?|results?|evals?)(?:/|$)", path or "")
            else "checkpoint"
        )
        output(declared_out, kind, "setup.output_dir")

    data_outputs = set()
    for i, data in enumerate(setup.get("data") or []):
        if not isinstance(data, dict):
            continue
        prefix = f"setup.data[{i}]"
        data_segments, _ = _segments(data.get("build_command"))
        for segment in data_segments:
            for flag, values in _flags(segment):
                for raw in _values(values):
                    if flag in OUTPUT_FLAGS:
                        data_outputs.add(
                            output(raw, "generated_data", prefix + ".build_command.output")
                        )
                    elif flag in WEIGHT_FLAGS:
                        reference(raw, "generated_data", prefix + ".build_command.generator")
                    elif flag in DATA_FLAGS:
                        reference(raw, "generated_data", prefix + ".build_command.input")
                    elif flag in CONFIG_FLAGS:
                        reference(raw, "configuration", prefix + ".build_command.config")
        if isinstance(data.get("path"), str) and data.get("built_by"):
            data_outputs.add(output(data["path"], "generated_data", prefix + ".declared_built_by"))
        for ref in card_refs(data.get("source")):
            reference(ref, "generated_data", prefix + ".source_card_reference", card_only=True)
        raw_path = data.get("path")
        if isinstance(raw_path, str):
            reference(
                raw_path,
                "generated_data",
                prefix + ".path",
                internal=_artifact(raw_path, cwd) in data_outputs,
            )

    segments, command_warnings = _segments(command.get("argv", command.get("command")))
    warnings.extend(command_warnings)
    prior_command_outputs = set()
    command_weight_refs = []
    merge_seen = False
    for number, segment in enumerate(segments):
        merge = _is_merge(segment, family)
        merge_seen |= merge
        produced = set()
        for flag, values in _flags(segment):
            kind = None
            if flag in OUTPUT_FLAGS:
                if not _is_eval(segment):
                    for raw in _values(values):
                        produced.add(
                            output(raw, "checkpoint", f"{command_field}.segment[{number}].--{flag}")
                        )
                continue
            if flag in WEIGHT_FLAGS or (merge and flag in MERGE_FLAGS) or flag == "src":
                kind = "weights"
            elif flag in CONFIG_FLAGS:
                kind = "configuration"
            elif flag in DATA_FLAGS:
                kind = "generated_data"
            if kind is None:
                continue
            for raw in _values(values):
                coefficient = None
                if merge:
                    weighted = re.fullmatch(r"(.+):([+-]?(?:\d+(?:\.\d*)?|\.\d+))", raw)
                    if weighted:
                        raw, coefficient = weighted.groups()
                if posixpath.basename(raw) in CONFIG_NAMES:
                    kind = "configuration"
                internal = _artifact(raw, cwd) in prior_command_outputs
                if kind == "generated_data":
                    internal |= _artifact(raw, cwd) in data_outputs
                reference(
                    raw,
                    kind,
                    f"{command_field}.segment[{number}].--{flag}",
                    internal=internal,
                    coefficient=coefficient,
                )
                if kind == "weights" and not internal:
                    command_weight_refs.append(raw)
        # Inline Python is parsed as syntax, never executed. Only literal config
        # file paths are extracted; constants, prints, and metric values are ignored.
        if "-c" in segment and any("python" in t for t in segment[:2]):
            try:
                code = segment[segment.index("-c") + 1]
                tree = ast.parse(code)
                for value in ast.walk(tree):
                    if (
                        isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                        and posixpath.basename(value.value) in CONFIG_NAMES
                    ):
                        reference(
                            value.value,
                            "configuration",
                            f"{command_field}.inline_python_config_literal",
                            internal=_artifact(posixpath.dirname(value.value), cwd)
                            in prior_command_outputs,
                        )
            except (IndexError, SyntaxError):
                warnings.append("inline_python_not_statically_parsed")
        prior_command_outputs.update(produced)

    node["is_merge_declared"] = merge_seen or bool(
        re.search(r"merge|soup|averag", family, re.IGNORECASE)
    )
    if not command_weight_refs:
        path = parent.get("path")
        if isinstance(path, str) and path:
            reference(path, "weights", "setup.parent_checkpoint.path")
        elif node["origin_refs"]:
            for ref in node["origin_refs"]:
                reference(ref, "weights", "setup.parent_checkpoint.origin", card_only=True)
        elif node["origin_declares_base"] and base_model:
            reference(base_model, "weights", "setup.parent_checkpoint.origin_base")
        else:
            reference("[missing parent declaration]", "weights", "setup.parent_checkpoint.missing")
    else:
        node["warnings"].append("command_weight_inputs_take_priority_over_origin")
    if re.search(r"decod|config|packag", family, re.IGNORECASE):
        for ref in list(node["references"]):
            if ref["kind"] == "weights" and not ref["internal"]:
                node["references"].append(
                    {
                        **ref,
                        "kind": "configuration",
                        "evidence": "inherited_checkpoint_configuration",
                    }
                )
    if node["is_merge_declared"] and not command_weight_refs:
        reference(
            "[unparsed merge ingredients]", "weights", "merge_ingredients_not_statically_parsed"
        )
    node["warnings"].append("declared_artifacts_not_content_verified")
    return node


def build_graph(rows):
    """All cards enter the graph; no label/eligibility-dependent filtering."""
    projected = {}
    for row in rows:
        if not re.fullmatch(r"(?:aime-)?r0-\d+", row["cell_id"]) or not re.fullmatch(
            r"exp-\d+", row["card_id"]
        ):
            raise ValueError("Invalid run/card identity")
        if (
            row["example_id"] != row["cell_id"] + "/" + row["card_id"]
            or row["example_id"] in projected
        ):
            raise ValueError("Duplicate or inconsistent example identity")
        projected[row["example_id"]] = _project(row)
    times = {key: _time(node["first_submitted_at"]) for key, node in projected.items()}
    aliases = defaultdict(list)
    for key, node in projected.items():
        for output in node["outputs"]:
            if output["kind"] != "evaluation_output" and not _dynamic(output["artifact"]):
                aliases[node["cell_id"]].append(
                    (output["artifact"], key, output["kind"], output["evidence"])
                )
    nodes = {}
    for key, node in sorted(projected.items()):
        edges = []
        for reference in node["references"]:
            edge = {
                **reference,
                "producer_id": None,
                "resolution_status": "unresolved",
                "artifact_variant": None,
                "content_verified": False,
                "unresolved_reason": None,
                "dependency_scope": "requires_producer_weights"
                if reference["kind"] == "weights"
                else "configuration_only"
                if reference["kind"] == "configuration"
                else "unresolved",
                "declared_or_candidate_edge": True,
            }
            raw, artifact = reference["reference"], reference["artifact"]
            if reference["internal"]:
                edge["resolution_status"] = "internal_declared_operation"
                edge["dependency_scope"] = "in_recipe_operation"
            elif _base_reference(raw, node["base_model"]) is not None:
                edge["resolution_status"] = "declared_base_model"
                edge.update(_base_reference(raw, node["base_model"]))
            elif reference["card_only"]:
                candidate = node["cell_id"] + "/" + raw
                edge["provisional_candidates"] = [candidate] if candidate in projected else []
                if candidate not in projected:
                    edge["unresolved_reason"] = "referenced_card_missing"
                elif not times[key] or not times[candidate] or times[candidate] >= times[key]:
                    edge["unresolved_reason"] = "referenced_card_not_strictly_earlier"
                else:
                    edge.update(
                        producer_id=candidate,
                        resolution_status="declared_card_reference_only",
                        artifact_variant="unspecified",
                        resolution_method="explicit_card_reference",
                    )
                if reference["kind"] == "generated_data":
                    edge["ambiguity_reason"] = (
                        "source_card_reference_does_not_distinguish_data_reuse_from_generator_weights"
                    )
            elif _dynamic(artifact) or raw.startswith(("[missing", "[unparsed")):
                edge["unresolved_reason"] = "missing_or_dynamic_artifact"
            else:
                matches = [
                    (path, producer, kind, field)
                    for path, producer, kind, field in aliases[node["cell_id"]]
                    if (artifact == path or artifact.startswith(path + "/"))
                    and (reference["kind"] != "weights" or kind == "checkpoint")
                    and (reference["kind"] != "configuration" or kind == "checkpoint")
                ]
                prior = [
                    m
                    for m in matches
                    if m[1] != key and times[key] and times[m[1]] and times[m[1]] < times[key]
                ]
                if prior:
                    longest = max(len(m[0]) for m in prior)
                    prior = [m for m in prior if len(m[0]) == longest]
                    latest = max(times[m[1]] for m in prior)
                    prior = [m for m in prior if times[m[1]] == latest]
                    producers = sorted({m[1] for m in prior})
                    if len(producers) == 1:
                        match = min(prior)
                        edge.update(
                            producer_id=producers[0],
                            resolution_status="time_qualified_declared_artifact",
                            artifact_variant=artifact[len(match[0]) :].lstrip("/")
                            or "declared_output_root",
                            matched_output_alias=match[0],
                            output_evidence=match[3],
                            resolution_method="longest_declared_prefix_then_latest_prior_proposal",
                        )
                        if reference["kind"] == "generated_data":
                            edge["dependency_scope"] = (
                                "data_builder_only"
                                if match[2] == "generated_data"
                                else "requires_producer_weights"
                            )
                    else:
                        edge.update(
                            unresolved_reason="ambiguous_same_time_producers",
                            provisional_candidates=producers,
                        )
                elif matches:
                    edge.update(
                        unresolved_reason="artifact_has_no_strictly_earlier_producer",
                        provisional_candidates=sorted({m[1] for m in matches if m[1] != key}),
                    )
                else:
                    candidates = [node["cell_id"] + "/" + ref for ref in card_refs(raw)]
                    edge.update(
                        unresolved_reason="artifact_not_declared",
                        provisional_candidates=[c for c in candidates if c in projected],
                    )
            # Identical references across source fields coalesce, but distinct
            # artifact variants never coalesce merely because producer IDs match.
            equivalent = next(
                (
                    e
                    for e in edges
                    if all(
                        e.get(f) == edge.get(f)
                        for f in (
                            "kind",
                            "artifact",
                            "reference",
                            "producer_id",
                            "resolution_status",
                            "merge_coefficient",
                        )
                    )
                ),
                None,
            )
            if equivalent is not None:
                equivalent["evidence_fields"].append(edge["evidence"])
            else:
                edge["evidence_fields"] = [edge["evidence"]]
                edges.append(edge)
        weights = {
            e["producer_id"].split("/")[-1]
            for e in edges
            if e["kind"] == "weights" and e["producer_id"]
        }
        warnings = list(node["warnings"])
        if node["origin_refs"] and weights and set(node["origin_refs"]) != weights:
            warnings.append("origin_reference_disagrees_with_resolved_weight_inputs")
        crosscheck = None
        expected = MERGE_PARENTS.get(node["cell_id"], {}).get(int(node["card_id"].split("-")[1]))
        if expected is not None:
            expected_ids = {f"exp-{n:02d}" for n in expected}
            crosscheck = {
                "expected_card_ids": sorted(expected_ids),
                "resolved_weight_card_ids": sorted(weights),
                "matches": expected_ids == weights,
                "policy": "crosscheck_only_never_adds_edges",
            }
            if expected_ids != weights:
                warnings.append("legacy_merge_crosscheck_disagrees_or_incomplete")
        nodes[key] = {k: v for k, v in node.items() if k not in {"references", "warnings"}}
        nodes[key].update(
            parents=edges, warnings=sorted(set(warnings)), merge_crosscheck=crosscheck
        )
    children = defaultdict(set)
    for key, node in nodes.items():
        for edge in node["parents"]:
            if edge["producer_id"]:
                children[edge["producer_id"]].add(key)

    def closure(key, visiting=None):
        visiting = set() if visiting is None else visiting
        if key in visiting:
            raise ValueError("Cycle in declared lineage")
        result = []
        for parent in sorted({e["producer_id"] for e in nodes[key]["parents"] if e["producer_id"]}):
            for item in closure(parent, visiting | {key}):
                if item not in result:
                    result.append(item)
        return [*result, key]

    for key, node in nodes.items():
        node["topological_closure"] = closure(key)
        node["children"] = sorted(children[key])
        node["terminal_in_resolved_graph"] = not children[key]
        node["terminal_status_is_provisional"] = True
        node["closure_scope"] = (
            "card-level provenance only; data_builder_only never authorizes producer training"
        )
        node["full_recipe_execution_certified"] = False
        node["operation_expansion_required"] = any(
            edge["dependency_scope"] in {"data_builder_only", "unresolved"}
            for ancestor in node["topological_closure"]
            for edge in nodes[ancestor]["parents"]
        )
        node["closure_has_unresolved_dependencies"] = any(
            edge["resolution_status"] in {"unresolved", "declared_card_reference_only"}
            for ancestor in node["topological_closure"]
            for edge in nodes[ancestor]["parents"]
        )
    statuses = Counter(e["resolution_status"] for n in nodes.values() for e in n["parents"])
    return {
        "schema": "wm-declared-lineage-v1",
        "input_projection_sha256": _digest(projected),
        "nodes": nodes,
        "coverage": {
            "cards": len(nodes),
            "runs": len({n["cell_id"] for n in nodes.values()}),
            "edges_by_status": dict(sorted(statuses.items())),
            "terminal_nodes_in_resolved_graph": sum(
                n["terminal_in_resolved_graph"] for n in nodes.values()
            ),
            "closures_without_unresolved_declared_dependencies": sum(
                not n["closure_has_unresolved_dependencies"] for n in nodes.values()
            ),
        },
        "limits": [
            "Declared/inferred, not verified artifact identity or execution.",
            "All input cards precede eligibility filtering; terminal flags select no targets.",
            "Origin/data prose contributes explicit card IDs only, never score values.",
            "Missing aliases, dynamic shell control flow, and unknown variants remain unresolved.",
            "Hash-null declarations remain usable but never count as content-verified.",
        ],
    }


def build_artifact(inventory_path, output_dir):
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a new immutable lineage output directory")
    raw = Path(inventory_path).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    graph = build_graph(rows)
    output.mkdir(parents=True)
    for name, value in [
        ("graph.json", graph),
        (
            "provenance.json",
            {
                "inventory_path": str(Path(inventory_path).resolve()),
                "inventory_sha256": hashlib.sha256(raw).hexdigest(),
                "graph_sha256": _digest(graph),
                "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "legacy_merge_crosscheck_sha256": _digest(MERGE_PARENTS),
                "label_values_used": False,
            },
        ),
    ]:
        with (output / name).open("x") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    graph = build_artifact(args.inventory, args.output_dir)
    print(json.dumps(graph["coverage"], indent=2))


if __name__ == "__main__":
    main()
