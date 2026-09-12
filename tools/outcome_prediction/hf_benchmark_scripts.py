"""Recover bounded PTB experiment scripts for the HF prediction benchmark.

Nothing from a trajectory is executed.  A script is selected at a recorded
launch, strictly before its first checkpoint archive, using successful observed
file versions and a conservative intervening-write audit.  Final snapshots,
card prose, measurements, and arbitrary earlier experiment prefixes are never
model inputs.  ``eligible`` means this recorded recipe is sufficiently bound;
it is not a certificate of runtime bytes, dependency completeness, or replay.
"""

from __future__ import annotations

import ast
import hashlib
import json
import posixpath
import re
import shlex
from collections import defaultdict
from pathlib import Path

from tools.outcome_prediction.prefix_dataset import (
    OUTCOME_LITERAL,
    QUARANTINE,
    build_code_events,
    declared_code_names,
    load_cards,
)
from tools.outcome_prediction.rpm_code_provenance import (
    TraceIndex,
    explicit_shell_mutations,
    normalize_path,
    timestamp,
)
from tools.outcome_prediction.wm_checkpoint_inputs import resolve_inputs
from tools.outcome_prediction.wm_checkpoint_paths import resolve_target_binding
from tools.outcome_prediction.wm_code_features import _script_index

LIMITATIONS = [
    "Recorded prelaunch versions and archive paths do not prove executed or archived weight bytes.",
    "Static mutation auditing cannot rule out indirect writes or missing trace events.",
    "External library versions, remote dataset contents and stochastic state are not fully reconstructed.",
    "Sanitization removes comments/docstrings and obvious diagnostic metric strings; it is not a semantic leakage proof.",
]
_UNSAFE = re.compile(r"[$`*?{}]|[\r\n]")
_ENV = re.compile(r"([A-Za-z_]\w*)=(.*)", re.DOTALL)
_PRIVATE_ENV = re.compile(r"(?i)(?:^|_)(?:token|secret|password|api_key|credential)(?:$|_)")
_OUTCOME_ENV = re.compile(r"(?i)accuracy|score|result|baseline|correct")
_ENV_MUTATORS = {"source", ".", "unset", "eval", "set", "read", "declare", "typeset", "local", "readonly", "trap"}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _literal(value):
    return isinstance(value, str) and bool(value) and not _UNSAFE.search(value)


def _json(path):
    return json.loads(Path(path).read_text())


def _card_command(card):
    setup = card.get("setup") or {}
    command = setup.get("command") or {}
    return command if isinstance(command, dict) else {}


def _segments(command):
    """Tokenize simple shell statements; exclude heredocs/control-flow programs."""
    # Parsing text inside a heredoc as shell would invent launches from source
    # strings. It is safe to miss a same-command launch here and report it.
    if not isinstance(command, str) or "<<" in command or re.search(
        r"(?m)^\s*(?:for|while|if|case|function)\b", command
    ):
        return []
    try:
        text = command.replace("\\\n", " ")
        text = re.sub(r"(?<!\S)([012])(?=[<>])", r"__fd_\1", text)
        lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|<>\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        pieces, chunk, previous = [], [], None
        for token in lexer:
            if token and all(c in ";&|\n" for c in token):
                if chunk:
                    pieces.append((chunk, previous, token))
                chunk, previous = [], token
            else:
                chunk.append(token)
        if chunk:
            pieces.append((chunk, previous, None))
        return pieces
    except ValueError:
        return []


def parse_launches(command, cwd):
    """Extract literal Python launches, without outputs, logs or shell programs.

    Supports cd, per-command env assignments, nohup and ordinary redirections.
    Output pipes to diagnostic readers are allowed. Input pipes, OR branches,
    dynamic expansion and shell control flow are unresolved.
    ``segment`` is audit metadata and must be stripped from model input.
    """
    launches, exported, environment_known = [], {}, True
    segments = _segments(command)
    for segment, (raw, preceding, following) in enumerate(segments):
        tokens = list(raw)
        head = list(tokens)
        while head and (_ENV.fullmatch(head[0]) or head[0] in {"env", "nohup"}):
            head.pop(0)
        if head and head[0] in _ENV_MUTATORS:
            environment_known = False
            continue
        if preceding == "||" or following == "||":
            # OR-dependent exports/cd cannot be reduced to a single known state.
            environment_known = False
            continue
        diagnostic_pipe = (
            following == "|" and segment + 1 < len(segments)
            and segments[segment + 1][0][:1] in [["tail"], ["head"], ["grep"], ["tee"]]
        )
        if preceding in {"|", "||"} or following == "||" or following == "|" and not diagnostic_pipe:
            continue
        if tokens[:1] == ["cd"]:
            cwd = normalize_path(tokens[1], cwd) if len(tokens) == 2 and _literal(tokens[1]) else None
            continue
        if tokens[:1] == ["export"]:
            for token in tokens[1:]:
                match = _ENV.fullmatch(token)
                if match and _literal(match[2]):
                    exported[match[1]] = match[2]
                else:
                    environment_known = False
            continue
        if tokens and all(_ENV.fullmatch(token) for token in tokens):
            environment_known = False
            continue
        if not cwd or not environment_known:
            continue
        env = dict(exported)
        while tokens and (tokens[0] in {"env", "nohup"} or _ENV.fullmatch(tokens[0])):
            token = tokens.pop(0)
            match = _ENV.fullmatch(token)
            if match:
                env[match[1]] = match[2]
        if tokens[:1] == ["timeout"] and len(tokens) > 2 and re.fullmatch(r"\d+[smhd]?", tokens[1]):
            tokens = tokens[2:]
        # Remove redirects with their destinations, including 2>&1. A digit
        # before a redirect is an FD only when adjacent in the original source;
        # shlex loses that adjacency, so only common stderr/stdout forms qualify.
        kept, i = [], 0
        while i < len(tokens):
            if tokens[i] in {">", ">>", "<", ">&", "&>"}:
                if kept and kept[-1] in {"__fd_0", "__fd_1", "__fd_2"}:
                    kept.pop()
                i += 2
            else:
                kept.append(tokens[i])
                i += 1
        if not kept or any(_UNSAFE.search(v) for v in kept):
            continue
        index = _script_index(kept)
        if index is None:
            continue
        if any(_UNSAFE.search(v) for v in env.values()):
            continue
        launches.append({
            "argv": kept,
            "cwd": cwd,
            "env": env,
            "script": normalize_path(kept[index], cwd),
            "segment": segment,
        })
    return launches


def _signature(command):
    """Match executable arguments exactly, normalizing only interpreter/path."""
    argv = command.get("argv") or []
    if isinstance(argv, str):
        try:
            argv = shlex.split(argv)
        except ValueError:
            return None
    if not isinstance(argv, list) or not all(isinstance(v, str) for v in argv):
        return None
    idx = _script_index(argv)
    cwd = command.get("cwd") or "/home/ben/task"
    if idx is None:
        # Some recorded commands use an explicit literal shell wrapper.
        if len(argv) == 3 and Path(argv[0]).name in {"bash", "sh"} and argv[1] in {"-c", "-lc"}:
            parsed = parse_launches(argv[2], cwd)
            return _signature(parsed[0]) if len(parsed) == 1 else None
        return None
    return (normalize_path(argv[idx], cwd), tuple(argv[idx + 1:]))


def _clean_launch(launch):
    clean = {key: launch[key] for key in ("argv", "cwd", "script")}
    clean["env"] = {k: v for k, v in launch["env"].items() if not _unsafe_env(k, v)}
    return clean


def _unsafe_env(key, value):
    return bool(_PRIVATE_ENV.search(key) or _OUTCOME_ENV.search(key) or OUTCOME_LITERAL.search(value))


def _output_argument(launch):
    argv = launch["argv"]
    for index, token in enumerate(argv):
        flag, _, value = token.partition("=")
        if flag in {"--out", "--output", "--output-dir", "--output_dir", "--save-dir"}:
            value = value or (argv[index + 1] if index + 1 < len(argv) else "")
            return normalize_path(value, launch["cwd"]) if _literal(value) else None
    return None


def _builder_entries(name, setup):
    return [
        d for d in setup.get("data") or [] if isinstance(d, dict)
        and name in declared_code_names({"setup": {"data": [d]}})
    ]


def _literal_builder_outputs(content, cwd):
    """Find literal/constant file writes without executing builder code.

    This provides declarations, not proof of conditional writes or file bytes.
    Unknown formatted paths and function/argparse-dependent output paths stay
    unresolved when no observed output CLI argument supplies their identity.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return set()
    values = defaultdict(set)

    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Call) and node.args and (
            isinstance(node.func, ast.Name) and node.func.id == "Path"
            or isinstance(node.func, ast.Attribute) and node.func.attr == "Path"
        ):
            return literal(node.args[0])
        return set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id].update(literal(node.value))
    # Conflicting reassignments do not establish one output path.
    values = defaultdict(set, {key: value for key, value in values.items() if len(value) == 1})
    outputs = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "open" and node.args:
            modes = literal(node.args[1]) if len(node.args) > 1 else next(
                (literal(kw.value) for kw in node.keywords if kw.arg == "mode"), {"r"}
            )
            if any("w" in mode or "a" in mode or "x" in mode for mode in modes):
                outputs.update(literal(node.args[0]))
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in {"write_text", "write_bytes"}:
                outputs.update(literal(node.func.value))
            elif node.func.attr in {"to_json", "to_csv", "to_parquet", "save_to_disk"} and node.args:
                outputs.update(literal(node.args[0]))
    return {normalize_path(path, cwd) for path in outputs if _literal(path)}


def _select_builder(candidates, name, setup, cwd):
    """Prefer the declared build argv, then its exact output dataset path."""
    entries = _builder_entries(name, setup)
    signatures, data_paths = set(), set()
    for entry in entries:
        command = entry.get("build_command")
        if isinstance(command, (list, str)):
            signature = _signature({"argv": command, "cwd": cwd})
            if signature:
                signatures.add(signature)
        if _literal(entry.get("path")):
            data_paths.add(normalize_path(entry["path"], cwd))
    exact = [launch for launch in candidates if _signature(launch) in signatures]
    output = [launch for launch in candidates if _output_argument(launch) in data_paths]
    selected = exact or output
    if not selected and not signatures and not data_paths and len({_signature(launch) for launch in candidates}) == 1:
        selected = candidates
    if not selected:
        return None
    return max(selected, key=lambda launch: (launch["time"], launch["line"]))


def _version(index, events_by_path, path, launch):
    options = [
        e for e in events_by_path.get(path, [])
        if e["time"] < launch["time"] and e["line"] < launch["line"]
    ]
    if not options:
        return None, "no_recorded_script_version_before_launch"
    selected = max(options, key=lambda e: (e["time"], e["line"]))
    start = selected["line"]
    recovered_calls = {
        e["evidence"].get("tool_use_id")
        for e in options if e["time"] <= selected["time"]
    }
    for call in index.calls:
        if not start < call["line"] <= launch["line"]:
            continue
        if call["name"] in {"Write", "Edit"} and normalize_path(
            call["input"].get("file_path"), index.cwd
        ) == path:
            return None, "intervening_script_mutation_unresolved"
        if call["name"] == "Bash" and call["id"] not in recovered_calls:
            raw = call["input"].get("command", "")
            if explicit_shell_mutations(raw, path, index.cwd):
                return None, "intervening_shell_script_mutation_unresolved"
    return selected, None


def _helper_paths(content, path, known_paths):
    """Only local modules that exist in observed evidence; not all imports."""
    found, missing_relative = set(), []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return found, ["entrypoint_python_parse_failed"]
    directory = posixpath.dirname(path)
    for node in ast.walk(tree):
        requests = []
        if isinstance(node, ast.Import):
            requests = [(name.name, 0) for name in node.names]
        elif isinstance(node, ast.ImportFrom):
            requests = [(node.module or "", node.level)]
        for module, level in requests:
            base = directory
            for _ in range(max(0, level - 1)):
                base = posixpath.dirname(base)
            stem = normalize_path(module.replace(".", "/"), base)
            options = {stem + ".py", posixpath.join(stem, "__init__.py")}
            existing = options & known_paths
            found.update(existing)
            if level and not existing:
                missing_relative.append("unobserved_relative_import:" + (module or "."))
    return found - {path}, missing_relative


def _model_source(scripts, launch, setup):
    return {
        "task": {"base_model": setup.get("base_model")},
        "plan": {"setup": {
            "base_model": setup.get("base_model"),
            "command": _clean_launch(launch),
            "output_dir": setup.get("output_dir"),
            "parent_checkpoint": setup.get("parent_checkpoint") or {},
            "method": {"family": (setup.get("method") or {}).get("family")},
        }},
        "code": [{
            "role": s["role"], "script_path": s["path"],
            "status": "reconstructed", "content": s["content"],
        } for s in scripts],
    }


def _resolve_files(source_root, fallback_root, relative):
    """Use one whole session source, never splice incompatible trace/cards.

    The caller must verify fallback hashes against its pinned HF inventory.
    This function records the actual source and hashes for independent checking.
    """
    here = source_root / relative
    if (here / "wm/records.jsonl").is_file() and (here / "solve_out_sanitized.txt").is_file():
        return here
    if fallback_root is not None:
        other = fallback_root / relative
        if (other / "wm/records.jsonl").is_file() and (other / "solve_out_sanitized.txt").is_file():
            return other
    return here


def _weight_writers(index, events_by_path, launches):
    """Audit explicit CLI and statically supported code save destinations."""
    result = []
    for launch in launches:
        destination = _output_argument(launch)
        paths = {destination} if destination else set()
        unknown = False
        version, _ = _version(index, events_by_path, launch["script"], launch)
        if version is not None:
            source = version["payload"]["content"]
            if re.search(r"save_pretrained|save_model|save_file|torch\.save", source):
                scripts = [{"path": launch["script"], "content": source, "role": "training"}]
                payload = _model_source(scripts, launch, {"output_dir": "/__hf_planned_output__"})
                bound = resolve_target_binding({
                    "model_input": payload, "first_stage": "plan",
                    "first_submitted_at": launch["time"].isoformat(),
                    "audit": {"reasons": [], "final_output_checkpoint": "/__hf_target_output__"},
                })
                paths.update(bound["candidate_save_paths"])
                unknown = any(
                    event.get("kind") == "weight_save" and event.get("path") is None
                    for event in bound["evidence"]
                )
        elif re.search(r"train|finetun|merge|soup|checkpoint", Path(launch["script"]).name, re.IGNORECASE):
            unknown = not destination
        for path in sorted(paths):
            result.append({"path": path, "at": launch["time"].isoformat(), "call_line": launch["line"]})
        if unknown:
            result.append({"path": None, "at": launch["time"].isoformat(),
                           "call_line": launch["line"], "unresolved_weight_write": True})
    return result


def _extract_card(session, card, index, events_by_path, launches, weight_writers, benchmark, base_model):
    checkpoint_id = session.name + "-" + card["card_id"]
    setup = (card["plan_card"].get("setup") or {})
    archive = card.get("archive")
    archived_card = card.get("archive_card") or {}
    archived_setup = archived_card.get("setup") or {}
    target_path = (archived_card.get("result") or {}).get("output_checkpoint")
    record = {
        "scripts": [], "launch": {}, "status": "candidate",
        "exclusion_reasons": [], "review_flags": [], "parent_checkpoint_ids": [],
        "session_id": session.name, "benchmark": benchmark, "base_model": base_model,
        "provenance": {
            "source_session": str(session), "trace_sha256": index.trace_sha256,
            "ledger_sha256": _hash(session / "wm/records.jsonl"),
            "record_files": [{
                "path": str(session / "wm/cards" / card["card_id"] / Path(entry["path"]).name),
                "sha256": _hash(session / "wm/cards" / card["card_id"] / Path(entry["path"]).name),
            } for entry in [card["plan"], *([archive] if archive else [])]],
            "first_archive_at": archive.get("at") if archive else None,
            "target_output_path": target_path,
            "limitations": LIMITATIONS,
            "script_evidence": [],
        },
    }
    reasons, flags = record["exclusion_reasons"], record["review_flags"]
    if checkpoint_id in QUARANTINE:
        reasons.append("known_recipe_label_quarantine")
    if not archive:
        reasons.append("no_recorded_checkpoint_archive")
        return record
    boundary = timestamp(archive["at"])
    declared = [_card_command(card["plan_card"])]
    if _card_command(archived_card) != declared[0]:
        declared.append(_card_command(archived_card))
    signatures = {s for c in declared if (s := _signature(c)) is not None}
    matching = [l for l in launches if l["time"] < boundary and _signature(l) in signatures]
    # The archive may retain a full training argv on a later evaluation-only
    # card. Only a launch after the previous archive can establish a new recipe;
    # the direct earlier recipe can instead be resolved as an alias below.
    if not matching:
        reasons.append("no_observed_launch_matching_recorded_argv")
        return record
    chosen = max(matching, key=lambda l: (l["time"], l["line"], l["segment"]))
    script = chosen["script"]
    if re.search(r"(?:^|[/_])(?:eval(?:uate)?|test|score|diagnos)[^/]*\.py$", script, re.IGNORECASE):
        reasons.append("evaluation_only_entrypoint_needs_underlying_training_recipe")
        return record
    if len(matching) > 1:
        # Successful repeated identical launches can resume/overwrite weights;
        # distinguish recorded failed attempts from an unresolved multi-launch run.
        completed = [l for l in matching if l.get("result") and not l["result"].get("is_error")]
        if len(completed) > 1:
            reasons.append("multiple_matching_training_launches")
    if (chosen.get("result") or {}).get("is_error"):
        reasons.append("training_launch_reported_tool_error")
    record["launch"] = _clean_launch(chosen)
    if any(_unsafe_env(k, v) for k, v in chosen["env"].items()):
        reasons.append("sensitive_or_outcome_environment_requires_review")
    record["provenance"]["launch_evidence"] = {
        "tool_use_id": chosen["id"], "call_line": chosen["line"],
        "at": chosen["time"].isoformat(), "segment": chosen["segment"],
        "binding": "recorded argv exact match after interpreter/path normalization",
    }
    destination = _output_argument(chosen)
    if destination and any(
        other["time"] < boundary
        and (other["time"], other["line"], other["segment"]) > (chosen["time"], chosen["line"], chosen["segment"])
        and _output_argument(other) == destination
        for other in launches
    ):
        reasons.append("output_path_reused_by_later_launch_before_archive")
    target_normalized = normalize_path(target_path, chosen["cwd"])
    for call in index.calls:
        if call["name"] != "Bash" or not chosen["time"] < call["time"] < boundary:
            continue
        if target_normalized and explicit_shell_mutations(
            call["input"].get("command", ""), target_normalized, index.cwd
        ):
            reasons.append("explicit_checkpoint_path_mutation_before_archive")
            break
    code_queue = [(script, "training", chosen)]
    selected_keys = set()
    builder_launches = []
    declarations = declared_code_names(card["plan_card"])
    for entry in setup.get("data") or []:
        if (
            isinstance(entry, dict) and entry.get("built_by")
            and str(entry["built_by"]).lower().strip() not in {"none", "null", "n/a"}
            and not declared_code_names({"setup": {"data": [entry]}})
        ):
            reasons.append("inline_data_builder_not_reconstructed")
    if declarations:
        for name in sorted(declarations):
            builder_path = normalize_path(name, chosen["cwd"])
            if builder_path == script or Path(builder_path).name == Path(script).name:
                continue
            candidates = [
                l for l in launches
                if l["time"] < chosen["time"] and l["line"] < chosen["line"]
                and (l["script"] == builder_path or name == Path(name).name and Path(l["script"]).name == name)
            ]
            if not candidates:
                reasons.append("declared_data_builder_launch_missing:" + name)
                continue
            builder = _select_builder(candidates, name, setup, chosen["cwd"])
            if builder is None:
                reasons.append("declared_data_builder_arguments_ambiguous:" + name)
                continue
            if (builder.get("result") or {}).get("is_error"):
                reasons.append("data_builder_launch_reported_tool_error:" + name)
            if not builder.get("result") or builder["result"]["time"] >= chosen["time"]:
                reasons.append("data_builder_tool_result_not_before_training:" + name)
            builder_output = _output_argument(builder)
            expected_data = {
                normalize_path(entry["path"], chosen["cwd"])
                for entry in _builder_entries(name, setup) if _literal(entry.get("path"))
            }
            if builder_output and expected_data and builder_output not in expected_data:
                reasons.append("declared_data_transform_not_reconstructed:" + name)
                record["provenance"].setdefault("unresolved_data_transforms", []).append({
                    "builder_script": name, "recorded_builder_output": builder_output,
                    "declared_training_data": sorted(expected_data),
                })
            if not builder_output:
                builder_version, _ = _version(index, events_by_path, builder["script"], builder)
                known_outputs = _literal_builder_outputs(
                    builder_version["payload"]["content"], builder["cwd"]
                ) if builder_version else set()
                if not expected_data or not expected_data <= known_outputs:
                    reasons.append("data_builder_output_binding_unresolved:" + name)
            if builder_output and any(
                other["time"] > builder["time"] and _output_argument(other) == builder_output
                for other in candidates
            ):
                reasons.append("data_builder_output_reused_before_training:" + name)
            if any(_unsafe_env(k, v) for k, v in builder["env"].items()):
                reasons.append("sensitive_or_outcome_builder_environment_requires_review")
            code_queue.append((builder["script"], "data_builder", builder))
            builder_launches.append(_clean_launch(builder))
    while code_queue:
        path, role, at_launch = code_queue.pop(0)
        key = (path, at_launch["line"])
        if key in selected_keys:
            continue
        selected_keys.add(key)
        selected, error = _version(index, events_by_path, path, at_launch)
        if error:
            reasons.append(error + ":" + path)
            continue
        content = selected["payload"]["content"]
        flags.extend(selected["flags"])
        if selected["flags"]:
            reasons.append("script_content_requires_review:" + path)
        record["scripts"].append({"path": path, "content": content, "role": role})
        record["provenance"]["script_evidence"].append({
            "path": path, "role": role, **selected["evidence"],
            "sanitized_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "launch_call_line": at_launch["line"],
        })
        helpers, errors = _helper_paths(content, path, set(events_by_path))
        reasons.extend(errors)
        for helper in sorted(helpers):
            code_queue.append((helper, "local_helper", at_launch))
    if builder_launches:
        record["launch"]["data_builders"] = builder_launches
    if not any(s["role"] == "training" for s in record["scripts"]):
        reasons.append("training_script_unavailable")
        return record
    # Use the matching declaration rather than mixing a changed plan with the
    # observed argv. Its prose/hyperparameter/result fields do not enter X.
    if _signature(_card_command(archived_card)) == _signature(chosen):
        setup = archived_setup
    setup = dict(setup, base_model=base_model)
    model_source = _model_source(record["scripts"], chosen, setup)
    binding_row = {
        "model_input": model_source,
        "first_stage": "plan", "first_submitted_at": chosen["time"].isoformat(),
        "audit": {"reasons": [], "final_output_checkpoint": target_path},
    }
    binding = resolve_target_binding(binding_row)
    record["provenance"]["target_binding"] = binding
    if binding["status"] not in {"exact", "planned_save_path"}:
        reasons.append("checkpoint_output_binding_unresolved")
    inputs = resolve_inputs(model_source)
    record["provenance"]["checkpoint_inputs"] = inputs
    input_paths = {item["path"] for item in inputs["inputs"] if "base_model" not in item}
    record["provenance"]["input_path_writers"] = [
        writer for writer in weight_writers if timestamp(writer["at"]) < chosen["time"]
        and (writer["path"] in input_paths or writer.get("unresolved_weight_write"))
    ]
    for call in index.calls:
        if call["name"] != "Bash" or call["time"] >= chosen["time"]:
            continue
        for path in sorted(input_paths):
            if explicit_shell_mutations(call["input"].get("command", ""), path, index.cwd):
                record["provenance"]["input_path_writers"].append({
                    "path": path, "at": call["time"].isoformat(), "call_line": call["line"],
                    "explicit_shell_mutation": True,
                })
    if inputs["status"] == "unresolved":
        reasons.append("input_checkpoint_binding_unresolved")
    elif inputs["status"] != "base":
        reasons.append("parent_training_recipe_not_yet_bound")
    flags.append("external_data_and_library_state_not_fully_reconstructed")
    record["exclusion_reasons"] = sorted(set(reasons))
    record["review_flags"] = sorted(set(flags))
    record["status"] = "candidate" if record["exclusion_reasons"] else "eligible"
    return record


def _attach_parents(records):
    """Attach only unique exact producer paths, recursively bounded to ancestors.

    No broad history is imported. Earlier branches/labels do not appear in X.
    Parents are composed only when they themselves passed all recipe gates.
    """
    for checkpoint_id, record in records.items():
        inputs = record["provenance"].get("checkpoint_inputs", {})
        current_time = record["provenance"].get("launch_evidence", {}).get("at")
        unresolved = False
        for item in inputs.get("inputs", []):
            if "base_model" in item:
                continue
            candidates = []
            for parent_id, parent in records.items():
                if parent_id == checkpoint_id or parent["session_id"] != record["session_id"]:
                    continue
                parent_path = parent["provenance"].get("target_output_path")
                archived_at = parent["provenance"].get("first_archive_at")
                if (
                    isinstance(parent_path, str) and isinstance(current_time, str) and archived_at
                    and normalize_path(parent_path, record.get("launch", {}).get("cwd", "/home/ben/task")) == item["path"]
                    and timestamp(archived_at) < timestamp(current_time)
                ):
                    reused = any(
                        (writer["path"] == item["path"] or writer.get("unresolved_weight_write"))
                        and timestamp(archived_at) < timestamp(writer["at"]) < timestamp(current_time)
                        for writer in record["provenance"].get("input_path_writers", [])
                    )
                    if not reused:
                        candidates.append(parent_id)
            if len(candidates) == 1:
                record["parent_checkpoint_ids"].extend(candidates)
            else:
                unresolved = True
        record["parent_checkpoint_ids"] = sorted(set(record["parent_checkpoint_ids"]))
        if unresolved:
            record["exclusion_reasons"].append("parent_source_path_not_unique_or_unarchived")
    # Entries are insertion ordered by first submission, but recurse explicitly
    # because a declared archive can arrive late or the source order can differ.
    done, visiting = set(), set()

    def visit(checkpoint_id):
        if checkpoint_id in done:
            return
        record = records[checkpoint_id]
        if checkpoint_id in visiting:
            record["exclusion_reasons"].append("cyclic_parent_recipe")
            return
        visiting.add(checkpoint_id)
        parents = record["parent_checkpoint_ids"]
        for parent in parents:
            visit(parent)
        reasons = record["exclusion_reasons"]
        if (
            parents and "parent_training_recipe_not_yet_bound" in reasons
            and "parent_source_path_not_unique_or_unarchived" not in reasons
            and all(records[p]["status"] == "eligible" for p in parents)
        ):
            reasons.remove("parent_training_recipe_not_yet_bound")
            # Versioned steps avoid confusing two historical contents of one
            # filename. Ancestor code and launches carry no scores or prose.
            record["launch"]["parent_recipes"] = [
                {"scripts": records[p]["scripts"], "launch": records[p]["launch"]}
                for p in parents
            ]
        record["exclusion_reasons"] = sorted(set(reasons))
        record["status"] = "candidate" if reasons else "eligible"
        visiting.remove(checkpoint_id)
        done.add(checkpoint_id)

    for checkpoint_id in records:
        visit(checkpoint_id)


def extract_ptb_scripts(source_root: Path, *, fallback_root: Path | None = None) -> dict:
    """Return all archived PTB cards, with usable inputs and explicit exclusions.

    An incomplete downloaded session may be read wholly from ``fallback_root``;
    callers must verify those file hashes against the pinned remote inventory.
    Fingerprints and source paths are returned outside the model input.
    """
    source_root = Path(source_root)
    fallback_root = Path(fallback_root) if fallback_root is not None else None
    sessions = set()
    for root in [source_root, fallback_root]:
        if root and (root / "cells").is_dir():
            sessions.update(p.name for p in (root / "cells").iterdir() if p.is_dir())
    records = {}
    for name in sorted(sessions):
        session = _resolve_files(source_root, fallback_root, Path("cells") / name)
        if not (session / "wm/records.jsonl").is_file() or not (session / "solve_out_sanitized.txt").is_file():
            continue
        try:
            cards = load_cards(session)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        if not cards:
            continue
        benchmark = "aime2025" if "aime" in name else "gsm8k"
        default_base = "Qwen/Qwen3-4B-Base" if benchmark == "aime2025" else "google/gemma-3-4b-pt"
        index = TraceIndex(session / "solve_out_sanitized.txt")
        # Passing no session_dir explicitly disables mutable card snapshots.
        events, _ = build_code_events(index, cards, session_dir=None)
        events_by_path = defaultdict(list)
        for event in events:
            if event["path"]:
                events_by_path[event["path"]].append(event)
        launches = []
        for call in index.calls:
            if call["name"] == "Bash":
                for launch in parse_launches(call["input"].get("command", ""), index.cwd):
                    launches.append({**call, **launch})
        weight_writers = _weight_writers(index, events_by_path, launches)
        for card in cards:
            base = (card["plan_card"].get("setup") or {}).get("base_model") or default_base
            records[name + "-" + card["card_id"]] = _extract_card(
                session, card, index, events_by_path, launches, weight_writers, benchmark, base
            )
    _attach_parents(records)
    return records
