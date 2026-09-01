"""Condition-scoped release evidence for the local production controller.

This module deliberately contains no scheduler, node, credential, or host-path
policy.  A site controller supplies its untracked study lock and persists the
returned gate document.  Enabling a condition requires a smoke from that exact
condition.  C1 and C2 additionally bind successful runs of the committed raw
corpus validator over both production scopes.

The helpers are side-effect free except for reading smoke/corpus artifacts.
They never write a release gate or submit work.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rollout import validate_study_corpus as corpus_validator

SCHEMA = "awm-production-release-gate-v3"
CONDITIONS = frozenset({"c1", "c2", "c3"})
RAW_CONDITIONS = frozenset({"c1", "c2"})
WMA_CONDITIONS = frozenset({"c2", "c3"})
SCOPES: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "train,test": ("train", "test"),
}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
MAX_JSON_BYTES = 16 * 1024 * 1024


class ReleaseGateError(RuntimeError):
    """Release evidence is missing, stale, or inconsistent."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_object(path: Path) -> dict[str, Any]:
    if not _regular(path):
        raise ReleaseGateError(f"required regular file is missing or linked: {path}")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ReleaseGateError(f"JSON evidence exceeds the release-gate limit: {path}")
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_object_without_duplicates)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise ReleaseGateError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ReleaseGateError(f"JSON top level is not an object: {path}")
    return value


def _validate_lock(lock: dict[str, Any]) -> None:
    try:
        harness_commit = lock["harness"]["commit"]
        ptb_commit = lock["posttrainbench"]["commit"]
        ptb_surface = lock["posttrainbench"]["surface_manifest_sha256"]
        base = lock["base_model"]
        inputs = lock["inputs"]
        claude = lock["claude"]
    except (KeyError, TypeError) as exc:
        raise ReleaseGateError("study lock lacks release-evidence fields") from exc
    if not isinstance(harness_commit, str) or not HEX40.fullmatch(harness_commit):
        raise ReleaseGateError("study lock has an invalid harness commit")
    if not isinstance(ptb_commit, str) or not HEX40.fullmatch(ptb_commit):
        raise ReleaseGateError("study lock has an invalid PTB commit")
    if not isinstance(ptb_surface, str) or not HEX64.fullmatch(ptb_surface):
        raise ReleaseGateError("study lock has an invalid PTB surface digest")
    if (
        not isinstance(base, dict)
        or not isinstance(base.get("id"), str)
        or not base["id"]
        or not isinstance(base.get("revision"), str)
        or not HEX40.fullmatch(base["revision"])
        or not isinstance(base.get("manifest_sha256"), str)
        or not HEX64.fullmatch(base["manifest_sha256"])
    ):
        raise ReleaseGateError("study lock has an invalid base-model contract")
    if not isinstance(inputs, dict) or set(inputs) != set(SCOPES):
        raise ReleaseGateError("study lock must pin exactly train and train,test inputs")
    for scope, row in inputs.items():
        if not isinstance(row, dict):
            raise ReleaseGateError(f"study input is not an object: {scope}")
        for key in ("prior_manifest_sha256", "card_manifest_sha256"):
            if not isinstance(row.get(key), str) or not HEX64.fullmatch(row[key]):
                raise ReleaseGateError(f"study input has an invalid {key}: {scope}")
        for key in ("prior_run_count", "card_count"):
            count = row.get(key)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ReleaseGateError(f"study input has an invalid {key}: {scope}")
        if not isinstance(row.get("prior_runs"), str) or not row["prior_runs"]:
            raise ReleaseGateError(f"study input has no raw corpus path: {scope}")
    aliases = claude.get("scientist_aliases") if isinstance(claude, dict) else None
    if (
        not isinstance(aliases, list)
        or not aliases
        or any(not isinstance(alias, str) or not alias for alias in aliases)
        or len(set(aliases)) != len(aliases)
    ):
        raise ReleaseGateError("study lock has an invalid scientist alias inventory")
    if (
        not isinstance(claude.get("cli_version"), str)
        or not claude["cli_version"]
        or not isinstance(claude.get("cli_version_output"), str)
        or not claude["cli_version_output"]
    ):
        raise ReleaseGateError("study lock has an invalid Claude CLI contract")


def _validator_sha256() -> str:
    path = Path(corpus_validator.__file__).resolve()
    if not _regular(path):
        raise ReleaseGateError("committed corpus validator is missing or linked")
    return _sha256_file(path)


def _input_contract(lock: dict[str, Any], condition: str, scope: str) -> tuple[str, int]:
    row = lock["inputs"][scope]
    if condition in RAW_CONDITIONS:
        return row["prior_manifest_sha256"], row["prior_run_count"]
    return row["card_manifest_sha256"], row["card_count"]


def new_release_gate(lock: dict[str, Any]) -> dict[str, Any]:
    """Return a commit-bound gate with all conditions safely disabled."""

    _validate_lock(lock)
    return {
        "schema_version": SCHEMA,
        "harness_commit": lock["harness"]["commit"],
        "conditions": {
            condition: {
                "enabled": False,
                "smoke_result": None,
                "smoke_evidence_sha256": None,
            }
            for condition in sorted(CONDITIONS)
        },
    }


def _verify_prompt(task: Path) -> tuple[Path, Path]:
    prompt = task / "instruction.md"
    checksum = task / "instruction.sha256"
    if (
        not _regular(prompt)
        or prompt.stat().st_size < 1
        or not _regular(checksum)
        or checksum.stat().st_size > 256
    ):
        raise ReleaseGateError("smoke prompt or its checksum is missing, empty, or linked")
    expected = f"{_sha256_file(prompt)}  instruction.md"
    try:
        actual = checksum.read_text().strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ReleaseGateError("smoke prompt checksum is unreadable") from exc
    if actual != expected:
        raise ReleaseGateError("smoke prompt checksum does not match instruction.md")
    return prompt, checksum


def _verify_structural_final_model(path: Path, *, lock: dict[str, Any], study_input: Path) -> None:
    attestation = _load_object(path)
    expected_base = attestation.get("expected_base")
    final_model = attestation.get("final_model")
    weights = final_model.get("weights") if isinstance(final_model, dict) else None
    if (
        attestation.get("schema_version") != "awm-final-model-structural-attestation-v2"
        or attestation.get("status") != "passed"
        or attestation.get("scope") != "structural-declarative-only"
        or attestation.get("causal_training_lineage_proven") is not False
        or not isinstance(expected_base, dict)
        or expected_base.get("model_id") != lock["base_model"]["id"]
        or expected_base.get("revision") != lock["base_model"]["revision"]
        or not isinstance(final_model, dict)
        or final_model.get("path_within_task") != "final_model"
        or not isinstance(weights, dict)
        or weights.get("tensor_topology_match") is not True
        or not isinstance(weights.get("topology"), str)
        or not weights["topology"]
        or not isinstance(weights.get("base_topology"), str)
        or not weights["base_topology"]
        or isinstance(weights.get("shard_count"), bool)
        or not isinstance(weights.get("shard_count"), int)
        or weights["shard_count"] < 1
        or isinstance(weights.get("tensor_count"), bool)
        or not isinstance(weights.get("tensor_count"), int)
        or weights["tensor_count"] < 1
        or isinstance(weights.get("bytes"), bool)
        or not isinstance(weights.get("bytes"), int)
        or weights["bytes"] < 1
        or isinstance(weights.get("tensor_payload_bytes"), bool)
        or not isinstance(weights.get("tensor_payload_bytes"), int)
        or weights["tensor_payload_bytes"] < 1
        or attestation.get("study_input_sha256") != _sha256_file(study_input)
    ):
        raise ReleaseGateError("smoke final-model topology attestation is invalid")


def verify_smoke_result(result: Path, lock: dict[str, Any], *, condition: str) -> dict[str, Any]:
    """Verify one exact condition-specific one-hour smoke result."""

    _validate_lock(lock)
    if condition not in CONDITIONS:
        raise ReleaseGateError("condition must be c1, c2, or c3")
    if result.is_symlink() or not result.is_dir():
        raise ReleaseGateError("smoke result must be a real directory")
    result = result.resolve()
    task = result / "task"
    if task.is_symlink() or not task.is_dir():
        raise ReleaseGateError("smoke task directory is missing or linked")
    prompt, prompt_checksum = _verify_prompt(task)
    study_input = task / "study-input.json"
    study = _load_object(study_input)
    if study.get("condition") != condition:
        raise ReleaseGateError("smoke result belongs to a different condition")
    final_attestation_name = (
        "c1-final-model-attestation.json"
        if condition == "c1"
        else "wma-final-model-attestation.json"
    )
    required = {
        "solve_exit": result / "solve_exit_code.txt",
        "metrics": result / "metrics.json",
        "study_input": study_input,
        "base_model": task / "base-model-attestation.json",
        "claude_cli": task / "claude-cli-attestation.json",
        "scientist_model": task / "scientist-model-attestation.json",
        "claude_exit": task / "claude-exit-code.txt",
        "instruction": prompt,
        "instruction_checksum": prompt_checksum,
        "final_model_attestation": task / final_attestation_name,
    }
    if condition in WMA_CONDITIONS:
        required["wma_session"] = task / "wma-session-attestation.json"
    if any(not _regular(path) for path in required.values()):
        raise ReleaseGateError("smoke result lacks a required regular attestation file")
    if required["solve_exit"].read_text().strip() != "0":
        raise ReleaseGateError("smoke solve exit is not zero")
    if required["claude_exit"].read_text().strip() != "0":
        raise ReleaseGateError("smoke Claude exit is not zero")

    metrics = _load_object(required["metrics"])
    accuracy = metrics.get("accuracy")
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
    ):
        raise ReleaseGateError("smoke official accuracy is not finite")

    expected_manifest, expected_count = _input_contract(lock, condition, "train")
    expected_kind = "raw" if condition in RAW_CONDITIONS else "cards"
    expected_study = {
        "condition": condition,
        "study_mode": "smoke",
        "num_hours": 1,
        "harness_commit": lock["harness"]["commit"],
        "ptb_commit": lock["posttrainbench"]["commit"],
        "ptb_surface_manifest_sha256": lock["posttrainbench"]["surface_manifest_sha256"],
        "kind": expected_kind,
        "schema_version": (
            corpus_validator.RAW_SCHEMA
            if condition in RAW_CONDITIONS
            else corpus_validator.CARD_SCHEMA
        ),
        "manifest_sha256": expected_manifest,
        "scope": ["train"],
        "validator_sha256": _validator_sha256(),
    }
    for field, expected in expected_study.items():
        if study.get(field) != expected:
            raise ReleaseGateError(f"smoke study-input mismatch: {field}")
    repetition = study.get("repetition")
    if isinstance(repetition, bool) or repetition not in (1, 2):
        raise ReleaseGateError("smoke study-input has no valid explicit repetition")
    count_field = "run_count" if condition in RAW_CONDITIONS else "card_count"
    if study.get(count_field) != expected_count:
        raise ReleaseGateError(f"smoke study-input mismatch: {count_field}")
    if condition in RAW_CONDITIONS:
        credential_scan = study.get("credential_scan")
        if (
            not isinstance(credential_scan, dict)
            or credential_scan.get("ruleset_version") != corpus_validator.RULESET_VERSION
            or type(credential_scan.get("finding_group_count")) is not int
            or credential_scan["finding_group_count"] != 0
            or isinstance(credential_scan.get("file_count"), bool)
            or not isinstance(credential_scan.get("file_count"), int)
            or credential_scan["file_count"] < 1
        ):
            raise ReleaseGateError("raw smoke lacks a zero-finding credential scan")

    base = _load_object(required["base_model"])
    if (
        base.get("schema_version") != "awm-base-model-cache-v1"
        or base.get("model_id") != lock["base_model"]["id"]
        or base.get("revision") != lock["base_model"]["revision"]
        or base.get("manifest_sha256") != lock["base_model"]["manifest_sha256"]
        or base.get("verification") != "full-content-hash"
        or study.get("base_model") != base
    ):
        raise ReleaseGateError("smoke base-model attestation is not the full locked cache")

    cli = _load_object(required["claude_cli"])
    if (
        cli.get("package") != "@anthropic-ai/claude-code"
        or cli.get("package_version") != lock["claude"]["cli_version"]
        or cli.get("actual_version_output") != lock["claude"]["cli_version_output"]
        or cli.get("expected_version_output") != lock["claude"]["cli_version_output"]
        or cli.get("update") != "pinned-install"
        or study.get("claude_cli") != cli
    ):
        raise ReleaseGateError("smoke Claude CLI attestation differs from the lock")

    model = _load_object(required["scientist_model"])
    alias = model.get("requested_alias")
    expected_model_id = model.get("expected_model_id")
    if (
        alias not in lock["claude"]["scientist_aliases"]
        or not isinstance(expected_model_id, str)
        or not expected_model_id
        or any(character.isspace() for character in expected_model_id)
        or model.get("reported_model_ids") != [expected_model_id]
        or model.get("reported_providers") != ["vertex"]
        or model.get("api_key_sources") != ["none"]
        or study.get("scientist_model") != model
    ):
        raise ReleaseGateError("smoke scientist model identity is not exact Vertex telemetry")

    if condition in WMA_CONDITIONS:
        wma = _load_object(required["wma_session"])
        lifecycle = wma.get("smoke_lifecycle")
        lineage = wma.get("base_lineage")
        if (
            not isinstance(lifecycle, dict)
            or not lifecycle.get("card_id")
            or not lifecycle.get("observation_id")
            or lifecycle.get("base_model") != lock["base_model"]["id"]
            or not isinstance(wma.get("adopted_card_ids"), list)
            or not wma["adopted_card_ids"]
            or isinstance(wma.get("successful_wma_call_count"), bool)
            or not isinstance(wma.get("successful_wma_call_count"), int)
            or wma["successful_wma_call_count"] < 1
            or not isinstance(lineage, dict)
            or lineage.get("base_model") != lock["base_model"]["id"]
            or study.get("wma_session") != wma
        ):
            raise ReleaseGateError("smoke WMA lifecycle attestation is incomplete")
    else:
        if "wma_session" in study or any(
            (task / name).exists() or (task / name).is_symlink()
            for name in (
                "wma-session-attestation.json",
                "wma-final-model-attestation.json",
            )
        ):
            raise ReleaseGateError("C1 smoke unexpectedly contains WMA evidence")

    _verify_structural_final_model(
        required["final_model_attestation"],
        lock=lock,
        study_input=required["study_input"],
    )
    final_model = result / "final_model"
    if final_model.is_symlink() or not final_model.is_dir() or not any(final_model.iterdir()):
        raise ReleaseGateError("smoke final_model is missing, linked, or empty")

    file_hashes = {name: _sha256_file(path) for name, path in sorted(required.items())}
    material = {
        "condition": condition,
        "scientist_alias": alias,
        "scientist_model_id": expected_model_id,
        "accuracy": float(accuracy),
        "input_kind": expected_kind,
        "input_manifest_sha256": expected_manifest,
        "files": file_hashes,
    }
    return {
        "result": str(result),
        **material,
        "evidence_sha256": _canonical_sha256(material),
    }


def validate_raw_release_inputs(lock: dict[str, Any]) -> dict[str, Any]:
    """Run the committed raw validator over both pinned production scopes."""

    _validate_lock(lock)
    validator_sha = _validator_sha256()
    records: dict[str, dict[str, Any]] = {}
    for scope, sides in SCOPES.items():
        row = lock["inputs"][scope]
        try:
            evidence = corpus_validator.validate_raw(
                Path(row["prior_runs"]), sides, row["prior_manifest_sha256"]
            )
        except (OSError, corpus_validator.ValidationError) as exc:
            raise ReleaseGateError(f"official raw corpus validation failed: {scope}") from exc
        credential_scan = evidence.get("credential_scan")
        if (
            evidence.get("kind") != "raw"
            or evidence.get("scope") != list(sides)
            or evidence.get("schema_version") != corpus_validator.RAW_SCHEMA
            or evidence.get("manifest_sha256") != row["prior_manifest_sha256"]
            or evidence.get("run_count") != row["prior_run_count"]
            or not isinstance(credential_scan, dict)
            or credential_scan.get("ruleset_version") != corpus_validator.RULESET_VERSION
            or type(credential_scan.get("finding_group_count")) is not int
            or credential_scan["finding_group_count"] != 0
        ):
            raise ReleaseGateError(
                f"official raw corpus evidence differs from the study lock: {scope}"
            )
        records[scope] = evidence
    material = {"validator_sha256": validator_sha, "scopes": records}
    return {**material, "evidence_sha256": _canonical_sha256(material)}


def _verify_raw_validation_document(value: Any, lock: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseGateError("enabled raw condition lacks corpus validation evidence")
    live = validate_raw_release_inputs(lock)
    if value != live:
        raise ReleaseGateError("raw corpus validation evidence is stale or changed")
    return live


def verify_release_gate(gate: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    """Validate every enabled condition and return current release evidence."""

    _validate_lock(lock)
    conditions = gate.get("conditions")
    if (
        gate.get("schema_version") != SCHEMA
        or gate.get("harness_commit") != lock["harness"]["commit"]
        or not isinstance(conditions, dict)
        or set(conditions) != CONDITIONS
    ):
        raise ReleaseGateError("release gate does not match the harness commit and conditions")

    enabled = []
    smoke_paths: set[str] = set()
    smoke_digests: set[str] = set()
    raw_validation: dict[str, Any] | None = None
    for condition in sorted(CONDITIONS):
        row = conditions[condition]
        if not isinstance(row, dict) or not isinstance(row.get("enabled"), bool):
            raise ReleaseGateError(f"invalid release gate entry for {condition}")
        if row["enabled"] is False:
            for field in (
                "smoke_result",
                "smoke_evidence_sha256",
                "raw_corpus_validation",
                "enabled_at",
            ):
                if row.get(field) is not None:
                    raise ReleaseGateError(
                        f"disabled {condition} release gate retains active evidence"
                    )
            continue
        smoke_result = row.get("smoke_result")
        expected_digest = row.get("smoke_evidence_sha256")
        if not isinstance(smoke_result, str) or not smoke_result:
            raise ReleaseGateError(f"enabled {condition} lacks a smoke result")
        if not isinstance(expected_digest, str) or not HEX64.fullmatch(expected_digest):
            raise ReleaseGateError(f"enabled {condition} lacks a smoke evidence digest")
        if not isinstance(row.get("enabled_at"), str) or not row["enabled_at"]:
            raise ReleaseGateError(f"enabled {condition} lacks an enable timestamp")
        evidence = verify_smoke_result(Path(smoke_result), lock, condition=condition)
        if evidence["evidence_sha256"] != expected_digest:
            raise ReleaseGateError(f"{condition} release-gate smoke evidence changed")
        if evidence["result"] in smoke_paths or expected_digest in smoke_digests:
            raise ReleaseGateError("one smoke result cannot release multiple conditions")
        smoke_paths.add(evidence["result"])
        smoke_digests.add(expected_digest)
        if condition in RAW_CONDITIONS:
            current = row.get("raw_corpus_validation")
            if raw_validation is None:
                raw_validation = _verify_raw_validation_document(current, lock)
            elif current != raw_validation:
                raise ReleaseGateError("enabled raw conditions have different corpus evidence")
        elif row.get("raw_corpus_validation") is not None:
            raise ReleaseGateError("C3 release gate unexpectedly contains raw corpus evidence")
        enabled.append(condition)
    return {
        "enabled_conditions": enabled,
        "raw_corpus_validation": raw_validation,
    }


def arm_condition(
    gate: dict[str, Any],
    lock: dict[str, Any],
    *,
    condition: str,
    smoke_result: Path,
    enabled_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a newly armed gate after all condition-specific checks pass."""

    if condition not in CONDITIONS:
        raise ReleaseGateError("condition must be c1, c2, or c3")
    current = verify_release_gate(gate, lock)
    evidence = verify_smoke_result(smoke_result, lock, condition=condition)
    raw_validation = None
    if condition in RAW_CONDITIONS:
        raw_validation = current["raw_corpus_validation"] or validate_raw_release_inputs(lock)
    updated = copy.deepcopy(gate)
    row = {
        "enabled": True,
        "smoke_result": evidence["result"],
        "smoke_evidence_sha256": evidence["evidence_sha256"],
        "enabled_at": enabled_at or _utcnow(),
    }
    if raw_validation is not None:
        row["raw_corpus_validation"] = raw_validation
    updated["conditions"][condition] = row
    verify_release_gate(updated, lock)
    return updated, evidence
