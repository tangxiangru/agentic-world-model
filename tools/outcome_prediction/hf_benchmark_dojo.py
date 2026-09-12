"""Bind Dojo's selected checkpoints to archived experiment scripts.

The unit is a run's explicitly exported best checkpoint, not every search step.
Development scores and search histories are provenance, never predictor input.
All reads are static: archived code is never executed. Eligibility here covers
script binding only; the benchmark builder separately validates serving and Y.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

from awm.credential_guard import scan_credential_bytes
from tools.outcome_prediction.prefix_dataset import MEASURED_LOG, OUTCOME_LITERAL, code_payload
from tools.outcome_prediction.wm_checkpoint_inputs import resolve_inputs

BASE_MODEL = "Qwen/Qwen3-4B-Base"
TRACK = "dojo_qwen_gsm8k"
_RUN_PATH = re.compile(r"dojo_ab_gsm8k/(rpm|traj)/seed(\d+)/run\.json$")
# Static reviews of two archived scripts the general interpreter cannot resolve.
# Any byte change invalidates the review. These certify the narrower weight-input
# observation only, not execution correctness or absence of semantic leakage.
_WEIGHT_SOURCE_REVIEWS = {
    "3dd579be2d41fb0df1f75f8db1d114badc7b4fd0d392e4f9973f757ef54c135d": {
        "reviewer": "Codex static source review, 2026-09-12",
        "resolution": "base_with_script_internal_intermediate",
        "evidence": "solution.py:60 loads fixed Qwen base; :120 saves ./phase1; :130 reloads it; :145 saves final checkpoint",
    },
    "a89482445af82cd1e9b0a4f023b967e9f5fbaf5176bcf9f17f66eac016c0022a": {
        "reviewer": "Codex static source review, 2026-09-12",
        "resolution": "base_in_both_try_except_branches",
        "evidence": "solution.py:401 and :411 both load literal Qwen/Qwen3-4B-Base; no other model load or resume",
    },
}


def _content_review(content: str, flags: list[str]) -> tuple[bool, list[str]]:
    """Distinguish outcomes from obvious runtime and mathematics syntax.

    Only the review scan is masked, never the experiment code. In particular,
    ``results.get('accuracy', 0.0)`` is a runtime lookup with a missing-value
    default, not an observed accuracy. A percentage without any metric marker
    is not sufficient evidence of an outcome (math data contains percentages).
    All remaining metric-bearing numbers retain the conservative review gate.
    """
    scan = content
    try:
        tree = ast.parse(content)
    except SyntaxError:
        tree = None
    if tree is not None:
        # AST columns use UTF-8 byte offsets. Byte spans also avoid accidentally
        # treating a quoted description of a get() call as executable syntax.
        raw = content.encode()
        offsets = [0]
        for line in raw.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        spans = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and len(node.args) == 2
                and not node.keywords
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "accuracy"
                and isinstance(node.args[1], ast.Constant)
                and type(node.args[1].value) in (int, float)
                and node.args[1].value == 0
            ):
                spans.append(
                    (
                        offsets[node.lineno - 1] + node.col_offset,
                        offsets[node.end_lineno - 1] + node.end_col_offset,
                    )
                )
        for start, stop in sorted(spans, reverse=True):
            raw = raw[:start] + b"runtime_metric_lookup()" + raw[stop:]
        scan = raw.decode()
    actual = [flag for flag in flags if "comments_not_removed" in flag]
    if OUTCOME_LITERAL.search(scan):
        actual.append("possible_outcome_literal_requires_content_review")
    if MEASURED_LOG.search(scan):
        actual.append("possible_measured_log_requires_content_review")
    if scan != content:
        actual.append("runtime_accuracy_zero_default_is_not_an_observation")
    if "percent_literal_requires_content_review" in flags:
        actual.append("percentage_syntax_present_semantics_not_exhaustively_reviewed")
    blocked = any(
        "requires_content_review" in flag or "comments_not_removed" in flag for flag in actual
    )
    return blocked, actual


def _json(path: Path):
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _record(root: Path, run_path: Path) -> tuple[str, dict]:
    path = _relative(run_path, root)
    match = _RUN_PATH.fullmatch(path)
    assert match is not None
    selector, seed = match.group(1), int(match.group(2))
    expected_id = f"abgsm8k-{selector}-s{seed:02d}"
    run = _json(run_path)
    config_path = run_path.parent / "dojo_config.json"
    config = _json(config_path) or {}
    reasons, flags = [], ["semantic_review_not_exhaustive", "best_dev_selected_checkpoint"]
    provenance = {"run_path": path, "config_path": _relative(config_path, root)}
    item = {
        "scripts": [],
        "launch": {"argv": ["python", "solution.py"], "cwd": None, "script": "solution.py"},
        "status": "candidate",
        "exclusion_reasons": reasons,
        "review_flags": flags,
        "provenance": provenance,
        "parent_checkpoint_ids": [],
        "session_id": f"dojo-{selector}-seed{seed:02d}",
        "benchmark": "gsm8k",
        "base_model": BASE_MODEL,
        "track": TRACK,
    }
    # The archived config contains a general script contract, but its concrete
    # run ID / cwd can be from an earlier launcher configuration. Neither its
    # agent prompt nor arbitrary runtime environment enters the returned input.
    config_text = json.dumps(config)
    if "python solution.py" not in config_text:
        flags.append("fixed_harness_launch_not_corroborated_by_config")
    declared_base = (config.get("task") or {}).get("base_model")
    if declared_base and declared_base != BASE_MODEL:
        reasons.append("dojo_config_base_model_conflict")
    elif not declared_base:
        flags.append("base_model_from_dojo_track_contract")
    if run is None:
        reasons.append("missing_or_invalid_run_metadata")
        return expected_id, item
    if run.get("rescore10_id") != expected_id:
        reasons.append("run_rescore_id_conflict_or_missing")
    if run.get("selector") != selector or run.get("seed") != seed:
        reasons.append("run_path_identity_conflict")
    provenance["exported_checkpoint"] = run.get("best_checkpoint_gs")
    exported = run.get("best_checkpoint_gs")
    if not isinstance(exported, str) or exported.rstrip("/").rsplit("/", 1)[-1] != expected_id:
        reasons.append("exported_checkpoint_identity_conflict_or_missing")
    configured_job = (config.get("metadata") or {}).get("slurm_id")
    if configured_job is not None and str(configured_job) != str(run.get("job")):
        flags.append("stale_config_working_directory_omitted")
    best = run.get("best")
    if (
        not isinstance(best, list)
        or len(best) != 3
        or not isinstance(best[1], str)
        or type(best[2]) is not int
        or best[2] < 1
    ):
        reasons.append("explicit_best_checkpoint_step_missing")
        return expected_id, item
    selected_checkpoint, selected_step = best[1], best[2]
    provenance.update(selected_step=selected_step, selected_checkpoint=selected_checkpoint)
    step_dir = run_path.parent / "artifacts" / f"step{selected_step:03d}"
    meta_path = step_dir / "meta.json"
    script_path = step_dir / "solution.py"
    provenance.update(
        meta_path=_relative(meta_path, root), script_path=_relative(script_path, root)
    )
    meta = _json(meta_path)
    if meta is None:
        reasons.append("selected_step_metadata_missing_or_invalid")
    else:
        if meta.get("step") != selected_step or meta.get("checkpoint") != selected_checkpoint:
            reasons.append("selected_checkpoint_step_metadata_conflict")
        if str(meta.get("run")) != str(run.get("job")):
            reasons.append("selected_step_run_identity_conflict")
        if meta.get("exit_code") != 0 or meta.get("is_buggy") is True:
            reasons.append("selected_checkpoint_execution_unsuccessful")
        # This is the code/search parent, not proof that the script consumes
        # parent weights. Do not create a false weight dependency from it.
        provenance["search_parent_step"] = meta.get("parent_step")
    try:
        raw = script_path.read_bytes()
        source = raw.decode("utf-8")
    except (OSError, UnicodeError):
        reasons.append("selected_step_script_missing_or_invalid")
        return expected_id, item
    source_sha = hashlib.sha256(raw).hexdigest()
    provenance["script_source_sha256"] = source_sha
    if meta is not None and meta.get("script_sha256") != source_sha:
        reasons.append("selected_script_checksum_conflict_or_missing")
    payload, content_flags = code_payload(source, "solution.py", "file_version")
    if scan_credential_bytes(payload["content"].encode(), path="solution.py"):
        reasons.append("script_credential_review_required")
        return expected_id, item
    provenance["script_input_sha256"] = hashlib.sha256(payload["content"].encode()).hexdigest()
    content_blocked, review_flags = _content_review(payload["content"], content_flags)
    flags.extend(review_flags)
    if content_blocked:
        reasons.append("script_content_review_required")
    item["scripts"].append(
        {"path": "solution.py", "content": payload["content"], "role": "experiment"}
    )
    # Require the identified weight inputs to be the fixed public base. A
    # search-parent pointer alone cannot supply a missing learned checkpoint.
    # A synthetic cwd gives relative paths stable identities; it is never
    # claimed to be the historical launch directory or exposed in X.
    weight_inputs = resolve_inputs(
        {
            "task": {"base_model": BASE_MODEL},
            "plan": {
                "setup": {
                    "command": {
                        "argv": ["python", "solution.py"],
                        "cwd": "/__dojo_workspace__",
                        "script": "solution.py",
                    }
                }
            },
            "code": [
                {
                    "script_path": "solution.py",
                    "status": "reconstructed",
                    "content": payload["content"],
                }
            ],
        }
    )
    provenance["weight_input_resolution"] = weight_inputs
    if weight_inputs["status"] != "base":
        if source_sha in _WEIGHT_SOURCE_REVIEWS:
            provenance["weight_input_source_review"] = {
                "script_source_sha256": source_sha,
                **_WEIGHT_SOURCE_REVIEWS[source_sha],
            }
            flags.append("weight_inputs_resolved_by_content_bound_static_review")
        else:
            reasons.append("script_weight_inputs_need_resolution")
    item["exclusion_reasons"] = sorted(set(reasons))
    item["review_flags"] = sorted(set(flags))
    item["status"] = "candidate" if reasons else "eligible"
    return expected_id, item


def extract_dojo_scripts(source_root: Path) -> dict[str, dict]:
    """Return all archived Dojo runs, retaining unresolved rows for audit.

    Labels are deliberately not opened here. A selected script can be bound
    even when its benchmark rescore is missing; label eligibility is separate.
    No scores, search plans, terminal output, or config prompts enter scripts.
    """
    root = Path(source_root)
    records = {}
    for run_path in sorted((root / "dojo_ab_gsm8k").glob("*/seed*/run.json")):
        if not _RUN_PATH.fullmatch(_relative(run_path, root)):
            continue
        checkpoint_id, record = _record(root, run_path)
        if checkpoint_id in records:
            raise ValueError(f"Duplicate Dojo checkpoint identity: {checkpoint_id}")
        records[checkpoint_id] = record
    return records
