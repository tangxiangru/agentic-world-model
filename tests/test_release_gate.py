from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from rollout import release_gate


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lock(tmp_path: Path) -> dict:
    return {
        "harness": {"commit": "a" * 40},
        "posttrainbench": {
            "commit": "b" * 40,
            "surface_manifest_sha256": "c" * 64,
        },
        "base_model": {
            "id": "google/gemma-3-4b-pt",
            "revision": "d" * 40,
            "manifest_sha256": "e" * 64,
        },
        "claude": {
            "cli_version": "2.1.251",
            "cli_version_output": "2.1.251 (Claude Code)",
            "scientist_aliases": [
                "claude-opus-4-6",
                "claude-opus-4-8",
                "claude-opus-5",
            ],
        },
        "inputs": {
            "train": {
                "prior_runs": str(tmp_path / "raw-train"),
                "prior_manifest_sha256": "1" * 64,
                "prior_run_count": 143,
                "card_manifest_sha256": "2" * 64,
                "card_count": 1580,
            },
            "train,test": {
                "prior_runs": str(tmp_path / "raw-train-test"),
                "prior_manifest_sha256": "3" * 64,
                "prior_run_count": 193,
                "card_manifest_sha256": "4" * 64,
                "card_count": 2030,
            },
        },
    }


def _disabled_gate(lock: dict) -> dict:
    return release_gate.new_release_gate(lock)


def _base_attestation(lock: dict) -> dict:
    return {
        "schema_version": "awm-base-model-cache-v1",
        "model_id": lock["base_model"]["id"],
        "revision": lock["base_model"]["revision"],
        "manifest_sha256": lock["base_model"]["manifest_sha256"],
        "verification": "full-content-hash",
    }


def _cli_attestation(lock: dict) -> dict:
    return {
        "actual_version_output": lock["claude"]["cli_version_output"],
        "expected_version_output": lock["claude"]["cli_version_output"],
        "package": "@anthropic-ai/claude-code",
        "package_version": lock["claude"]["cli_version"],
        "resolved_path": "/home/ben/.local/bin/claude",
        "update": "pinned-install",
        "version_record_sha256": "5" * 64,
    }


def _model_attestation(alias: str) -> dict:
    provider_id = f"vertex-{alias}"
    return {
        "api_key_sources": ["none"],
        "expected_model_id": provider_id,
        "reported_providers": ["vertex"],
        "reported_model_ids": [provider_id],
        "requested_alias": alias,
        "stream_sha256": "6" * 64,
    }


def _wma_attestation(lock: dict) -> dict:
    return {
        "adopted_card_ids": ["exp-smoke"],
        "event_count": 12,
        "events_sha256": "7" * 64,
        "successful_wma_call_count": 1,
        "smoke_lifecycle": {
            "card_id": "exp-smoke",
            "observation_id": "obs-smoke",
            "base_model": lock["base_model"]["id"],
        },
        "base_lineage": {
            "base_model": lock["base_model"]["id"],
            "final_card_id": "exp-smoke",
        },
    }


def _structural_attestation(lock: dict, study_input: Path) -> dict:
    return {
        "schema_version": "awm-final-model-structural-attestation-v2",
        "status": "passed",
        "scope": "structural-declarative-only",
        "causal_training_lineage_proven": False,
        "expected_base": {
            "model_id": lock["base_model"]["id"],
            "revision": lock["base_model"]["revision"],
        },
        "final_model": {
            "path_within_task": "final_model",
            "weights": {
                "tensor_topology_match": True,
                "topology": "candidate-topology",
                "base_topology": "base-topology",
                "shard_count": 2,
                "tensor_count": 10,
                "bytes": 1024,
                "tensor_payload_bytes": 1000,
            },
        },
        "study_input_sha256": _sha256(study_input),
    }


def _smoke_result(tmp_path: Path, lock: dict, condition: str, name: str | None = None) -> Path:
    result = tmp_path / (name or f"smoke-{condition}")
    task = result / "task"
    final_model = result / "final_model"
    task.mkdir(parents=True)
    final_model.mkdir()
    (result / "solve_exit_code.txt").write_text("0\n")
    (task / "claude-exit-code.txt").write_text("0\n")
    _write_json(result / "metrics.json", {"accuracy": 0.125})
    prompt = task / "instruction.md"
    prompt.write_text(f"exact {condition} smoke prompt\n")
    (task / "instruction.sha256").write_text(f"{_sha256(prompt)}  instruction.md\n")
    (final_model / "model.safetensors").write_bytes(b"model")

    base = _base_attestation(lock)
    cli = _cli_attestation(lock)
    model = _model_attestation("claude-opus-4-6")
    expected_kind = "raw" if condition in release_gate.RAW_CONDITIONS else "cards"
    expected_manifest = (
        lock["inputs"]["train"]["prior_manifest_sha256"]
        if expected_kind == "raw"
        else lock["inputs"]["train"]["card_manifest_sha256"]
    )
    study = {
        "condition": condition,
        "study_mode": "smoke",
        "num_hours": 1,
        "repetition": 1,
        "harness_commit": lock["harness"]["commit"],
        "ptb_commit": lock["posttrainbench"]["commit"],
        "ptb_surface_manifest_sha256": lock["posttrainbench"]["surface_manifest_sha256"],
        "kind": expected_kind,
        "manifest_sha256": expected_manifest,
        "scope": ["train"],
        "validator_sha256": release_gate._validator_sha256(),
        "base_model": base,
        "claude_cli": cli,
        "scientist_model": model,
    }
    if expected_kind == "raw":
        study.update(
            {
                "schema_version": release_gate.corpus_validator.RAW_SCHEMA,
                "run_count": lock["inputs"]["train"]["prior_run_count"],
                "credential_scan": {
                    "ruleset_version": release_gate.corpus_validator.RULESET_VERSION,
                    "file_count": 429,
                    "finding_group_count": 0,
                },
            }
        )
    else:
        study.update(
            {
                "schema_version": release_gate.corpus_validator.CARD_SCHEMA,
                "card_count": lock["inputs"]["train"]["card_count"],
            }
        )
    if condition in release_gate.WMA_CONDITIONS:
        wma = _wma_attestation(lock)
        study["wma_session"] = wma
        _write_json(task / "wma-session-attestation.json", wma)

    _write_json(task / "study-input.json", study)
    _write_json(task / "base-model-attestation.json", base)
    _write_json(task / "claude-cli-attestation.json", cli)
    _write_json(task / "scientist-model-attestation.json", model)
    final_name = (
        "c1-final-model-attestation.json"
        if condition == "c1"
        else "wma-final-model-attestation.json"
    )
    _write_json(
        task / final_name,
        _structural_attestation(lock, task / "study-input.json"),
    )
    return result


def _raw_validator(lock: dict):
    def validate(_root: Path, sides: tuple[str, ...], expected_sha: str) -> dict:
        scope = "train" if sides == ("train",) else "train,test"
        row = lock["inputs"][scope]
        assert expected_sha == row["prior_manifest_sha256"]
        return {
            "kind": "raw",
            "scope": list(sides),
            "manifest_sha256": expected_sha,
            "schema_version": release_gate.corpus_validator.RAW_SCHEMA,
            "split_id": "posttrainbench/gsm8k-gemma-holdout-v1",
            "dataset_revision": "8" * 40,
            "run_count": row["prior_run_count"],
            "credential_scan": {
                "ruleset_version": release_gate.corpus_validator.RULESET_VERSION,
                "file_count": row["prior_run_count"] * 3,
                "finding_group_count": 0,
            },
            "index_sha256": "9" * 64,
            "overview_sha256": "a" * 64,
            "readme_sha256": "b" * 64,
        }

    return validate


@pytest.mark.parametrize("condition", ["c1", "c2", "c3"])
def test_condition_specific_smoke_is_accepted(tmp_path: Path, condition: str) -> None:
    lock = _lock(tmp_path)
    result = _smoke_result(tmp_path, lock, condition)

    evidence = release_gate.verify_smoke_result(result, lock, condition=condition)

    assert evidence["condition"] == condition
    assert evidence["input_kind"] == (
        "raw" if condition in release_gate.RAW_CONDITIONS else "cards"
    )
    assert evidence["scientist_alias"] == "claude-opus-4-6"
    assert len(evidence["evidence_sha256"]) == 64


@pytest.mark.parametrize(
    ("actual", "requested"),
    [("c1", "c2"), ("c2", "c1"), ("c2", "c3"), ("c3", "c2")],
)
def test_smoke_cannot_release_a_different_condition(
    tmp_path: Path, actual: str, requested: str
) -> None:
    lock = _lock(tmp_path)
    result = _smoke_result(tmp_path, lock, actual)

    with pytest.raises(release_gate.ReleaseGateError, match="condition"):
        release_gate.verify_smoke_result(result, lock, condition=requested)


def test_c1_rejects_wma_evidence(tmp_path: Path) -> None:
    lock = _lock(tmp_path)
    result = _smoke_result(tmp_path, lock, "c1")
    task = result / "task"
    _write_json(task / "wma-session-attestation.json", {"unexpected": True})

    with pytest.raises(release_gate.ReleaseGateError, match="unexpectedly contains WMA"):
        release_gate.verify_smoke_result(result, lock, condition="c1")


def test_c2_requires_complete_wma_lifecycle(tmp_path: Path) -> None:
    lock = _lock(tmp_path)
    result = _smoke_result(tmp_path, lock, "c2")
    path = result / "task" / "wma-session-attestation.json"
    wma = json.loads(path.read_text())
    wma.pop("smoke_lifecycle")
    _write_json(path, wma)

    with pytest.raises(release_gate.ReleaseGateError, match="WMA lifecycle"):
        release_gate.verify_smoke_result(result, lock, condition="c2")


def test_raw_smoke_requires_zero_finding_scan(tmp_path: Path) -> None:
    lock = _lock(tmp_path)
    result = _smoke_result(tmp_path, lock, "c1")
    path = result / "task" / "study-input.json"
    study = json.loads(path.read_text())
    study["credential_scan"]["finding_group_count"] = 1
    _write_json(path, study)

    with pytest.raises(release_gate.ReleaseGateError, match="zero-finding"):
        release_gate.verify_smoke_result(result, lock, condition="c1")


def test_prompt_tampering_invalidates_smoke(tmp_path: Path) -> None:
    lock = _lock(tmp_path)
    result = _smoke_result(tmp_path, lock, "c3")
    (result / "task" / "instruction.md").write_text("changed\n")

    with pytest.raises(release_gate.ReleaseGateError, match="prompt checksum"):
        release_gate.verify_smoke_result(result, lock, condition="c3")


def test_raw_release_validation_runs_both_pinned_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _lock(tmp_path)
    calls = []
    validator = _raw_validator(lock)

    def recording(root: Path, sides: tuple[str, ...], expected_sha: str) -> dict:
        calls.append((root, sides, expected_sha))
        return validator(root, sides, expected_sha)

    monkeypatch.setattr(release_gate.corpus_validator, "validate_raw", recording)

    evidence = release_gate.validate_raw_release_inputs(lock)

    assert [call[1] for call in calls] == [("train",), ("train", "test")]
    assert set(evidence["scopes"]) == {"train", "train,test"}
    assert evidence["scopes"]["train"]["manifest_sha256"] == "1" * 64
    assert evidence["scopes"]["train,test"]["manifest_sha256"] == "3" * 64
    assert len(evidence["evidence_sha256"]) == 64


def test_raw_condition_gate_binds_smoke_and_live_corpora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _lock(tmp_path)
    gate = _disabled_gate(lock)
    result = _smoke_result(tmp_path, lock, "c1")
    monkeypatch.setattr(release_gate.corpus_validator, "validate_raw", _raw_validator(lock))

    armed, smoke = release_gate.arm_condition(
        gate,
        lock,
        condition="c1",
        smoke_result=result,
        enabled_at="2026-08-31T00:00:00Z",
    )
    verified = release_gate.verify_release_gate(armed, lock)

    assert gate["conditions"]["c1"]["enabled"] is False
    assert armed["conditions"]["c1"]["enabled"] is True
    assert armed["conditions"]["c1"]["smoke_evidence_sha256"] == smoke["evidence_sha256"]
    assert verified["enabled_conditions"] == ["c1"]
    assert verified["raw_corpus_validation"] == armed["conditions"]["c1"]["raw_corpus_validation"]


def test_raw_gate_fails_when_official_validation_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _lock(tmp_path)
    result = _smoke_result(tmp_path, lock, "c2")
    good = _raw_validator(lock)
    monkeypatch.setattr(release_gate.corpus_validator, "validate_raw", good)
    armed, _ = release_gate.arm_condition(
        _disabled_gate(lock), lock, condition="c2", smoke_result=result
    )

    def changed(root: Path, sides: tuple[str, ...], expected_sha: str) -> dict:
        evidence = good(root, sides, expected_sha)
        evidence["index_sha256"] = "f" * 64
        return evidence

    monkeypatch.setattr(release_gate.corpus_validator, "validate_raw", changed)
    with pytest.raises(release_gate.ReleaseGateError, match="stale or changed"):
        release_gate.verify_release_gate(armed, lock)


def test_raw_gate_rejects_missing_validation_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _lock(tmp_path)
    result = _smoke_result(tmp_path, lock, "c1")
    smoke = release_gate.verify_smoke_result(result, lock, condition="c1")
    gate = _disabled_gate(lock)
    gate["conditions"]["c1"] = {
        "enabled": True,
        "smoke_result": smoke["result"],
        "smoke_evidence_sha256": smoke["evidence_sha256"],
        "enabled_at": "2026-08-31T00:00:00Z",
    }
    monkeypatch.setattr(release_gate.corpus_validator, "validate_raw", _raw_validator(lock))

    with pytest.raises(release_gate.ReleaseGateError, match="lacks corpus"):
        release_gate.verify_release_gate(gate, lock)


def test_distinct_c1_and_c2_smokes_can_share_exact_raw_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _lock(tmp_path)
    monkeypatch.setattr(release_gate.corpus_validator, "validate_raw", _raw_validator(lock))
    gate, _ = release_gate.arm_condition(
        _disabled_gate(lock),
        lock,
        condition="c1",
        smoke_result=_smoke_result(tmp_path, lock, "c1"),
    )
    gate, _ = release_gate.arm_condition(
        gate,
        lock,
        condition="c2",
        smoke_result=_smoke_result(tmp_path, lock, "c2"),
    )

    verified = release_gate.verify_release_gate(gate, lock)

    assert verified["enabled_conditions"] == ["c1", "c2"]
    assert (
        gate["conditions"]["c1"]["raw_corpus_validation"]
        == gate["conditions"]["c2"]["raw_corpus_validation"]
    )


def test_one_smoke_path_cannot_release_two_conditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _lock(tmp_path)
    monkeypatch.setattr(release_gate.corpus_validator, "validate_raw", _raw_validator(lock))
    result = _smoke_result(tmp_path, lock, "c1")
    gate, _ = release_gate.arm_condition(
        _disabled_gate(lock), lock, condition="c1", smoke_result=result
    )
    gate["conditions"]["c2"] = copy.deepcopy(gate["conditions"]["c1"])

    with pytest.raises(release_gate.ReleaseGateError, match="condition"):
        release_gate.verify_release_gate(gate, lock)


def test_c3_gate_preserves_card_only_behavior_without_raw_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _lock(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("C3 gate must not inspect raw corpora")

    monkeypatch.setattr(release_gate.corpus_validator, "validate_raw", forbidden)
    gate, _ = release_gate.arm_condition(
        _disabled_gate(lock),
        lock,
        condition="c3",
        smoke_result=_smoke_result(tmp_path, lock, "c3"),
    )

    verified = release_gate.verify_release_gate(gate, lock)

    assert verified == {
        "enabled_conditions": ["c3"],
        "raw_corpus_validation": None,
    }
    assert "raw_corpus_validation" not in gate["conditions"]["c3"]


def test_disabled_raw_conditions_do_not_scan_corpora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _lock(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("disabled raw gates must not scan corpora")

    monkeypatch.setattr(release_gate.corpus_validator, "validate_raw", forbidden)

    assert release_gate.verify_release_gate(_disabled_gate(lock), lock) == {
        "enabled_conditions": [],
        "raw_corpus_validation": None,
    }


def test_gate_is_bound_to_exact_harness_commit(tmp_path: Path) -> None:
    lock = _lock(tmp_path)
    gate = _disabled_gate(lock)
    gate["harness_commit"] = "f" * 40

    with pytest.raises(release_gate.ReleaseGateError, match="harness commit"):
        release_gate.verify_release_gate(gate, lock)
