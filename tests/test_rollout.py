"""The study harness pieces that can be checked without a cluster."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEST_PRIOR_SPLIT = "posttrainbench/test-split-v1"
TEST_PRIOR_REVISION = "39d3fcd794df51c062c8bd3b7f8523ba707aaeb3"
TEST_PRIOR_DATASET = {
    "repo": "aisa-group/PostTrainBench-Trajectories",
    "repo_type": "dataset",
    "revision": TEST_PRIOR_REVISION,
    "catalog": "viewer_data/index.json",
    "catalog_sha256": "0" * 64,
}


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_prior_run(
    root: Path,
    run: str,
    accuracy: float = 0.5,
    *,
    revision: str = TEST_PRIOR_REVISION,
) -> Path:
    from awm.traj.fetch import _write_testable_download_metadata

    directory = root / run
    directory.mkdir(parents=True)
    (directory / "solve_out.txt").write_text("trace " * 100)
    (directory / "metrics.json").write_text(json.dumps({"accuracy": accuracy, "stderr": 0.02}))
    (directory / "time_taken.txt").write_text("09:58:00")
    for name in ("solve_out.txt", "metrics.json", "time_taken.txt"):
        _write_testable_download_metadata(root, f"{run}/{name}", revision)
    return directory


def _build_prior_runs(bpr, runs, raw: Path, out: Path, **kwargs):
    sides = kwargs.pop("sides", tuple(dict.fromkeys(side for _run, side in runs)))
    split_id = kwargs.pop("split_id", TEST_PRIOR_SPLIT)
    dataset = kwargs.pop("dataset", TEST_PRIOR_DATASET)
    return bpr.build(
        runs,
        raw,
        out,
        split_id=split_id,
        dataset=dataset,
        sides=sides,
        **kwargs,
    )


def test_build_prior_runs_copies_exact_declared_set_and_indexes(tmp_path: Path) -> None:
    bpr = _load(REPO / "tools" / "build_prior_runs.py")
    raw = tmp_path / "raw"
    runs = [("cfg_a/gsm8k_google_gemma-3-4b-pt_1", "test", 0.61),
            ("cfg_b/gsm8k_Qwen_Qwen3-4B-Base_2", "train", 0.74),
            ("cfg_b/gsm8k_Qwen_Qwen3-1.7B-Base_3", "train", 0.22)]
    for run, _side, acc in runs:
        d = _write_prior_run(raw, run, acc)
        (d / "solve_parsed.txt").write_text("optional upstream rendering")
        (d / "task").mkdir()
        (d / "task" / "train.py").write_text("print(1)")
    _write_prior_run(raw, "cfg_stale/gsm8k_Qwen_Qwen3-4B-Base_99")
    out = tmp_path / "prior_runs"
    summary = _build_prior_runs(bpr, [(r, s) for r, s, _ in runs], raw, out)
    assert summary["runs"] == 3 and summary["missing"] == []
    assert summary["by_side"] == {"train": 2, "test": 1}
    copied_run = out / "cfg_b/gsm8k_Qwen_Qwen3-4B-Base_2"
    assert {path.name for path in copied_run.iterdir()} == set(bpr.MANDATORY_FILES)
    assert not (copied_run / "task").exists()
    assert not (copied_run / "solve_parsed.txt").exists()
    assert not (out / "cfg_stale").exists()
    manifest_path = out / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == "awm-prior-runs-v1"
    assert manifest["split"] == {"id": TEST_PRIOR_SPLIT, "sides": ["test", "train"]}
    assert manifest["dataset"]["revision"] == TEST_PRIOR_REVISION
    assert manifest["run_count"] == 3
    assert manifest["file_scope"] == ["solve_out.txt", "metrics.json", "time_taken.txt"]
    for entry in manifest["runs"]:
        assert set(entry["files"]) == set(manifest["file_scope"])
        for file_record in entry["files"].values():
            assert file_record["bytes"] > 0
            assert len(file_record["sha256"]) == 64
    assert summary["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    index = [json.loads(l) for l in (out / "index.jsonl").read_text().splitlines()]
    assert index[0]["accuracy"] == 0.74 and index[0]["base_model"] == "Qwen/Qwen3-4B-Base"
    assert index[1]["base_model"] == "google/gemma-3-4b-pt"
    md = (out / "INDEX.md").read_text()
    assert "| 0.740 | Qwen/Qwen3-4B-Base | cfg_b |" in md and "/home/ben/prior_runs/cfg_a/" in md


@pytest.mark.parametrize("missing_name", ["solve_out.txt", "metrics.json", "time_taken.txt"])
def test_build_prior_runs_requires_every_mandatory_file(tmp_path: Path, missing_name: str) -> None:
    bpr = _load(REPO / "tools" / "build_prior_runs.py")
    raw = tmp_path / "raw"
    run = "cfg_a/gsm8k_google_gemma-3-4b-pt_1"
    _write_prior_run(raw, run).joinpath(missing_name).unlink()
    out = tmp_path / "prior_runs"
    with pytest.raises(bpr.PriorRunsError, match=missing_name):
        _build_prior_runs(bpr, [(run, "train")], raw, out)
    assert not out.exists()


def test_build_prior_runs_rejects_unproven_source_revision(tmp_path: Path) -> None:
    bpr = _load(REPO / "tools" / "build_prior_runs.py")
    raw = tmp_path / "raw"
    run = "cfg_a/gsm8k_google_gemma-3-4b-pt_1"
    _write_prior_run(raw, run, revision="4" * 40)
    out = tmp_path / "prior_runs"
    with pytest.raises(bpr.PriorRunsError, match="raw source provenance failed"):
        _build_prior_runs(bpr, [(run, "train")], raw, out)
    assert not out.exists()


def test_build_prior_runs_refuses_reuse_and_replace_removes_stale_runs(tmp_path: Path) -> None:
    bpr = _load(REPO / "tools" / "build_prior_runs.py")
    raw = tmp_path / "raw"
    run = "cfg_a/gsm8k_google_gemma-3-4b-pt_1"
    _write_prior_run(raw, run)
    out = tmp_path / "prior_runs"
    _build_prior_runs(bpr, [(run, "train")], raw, out)
    stale = "cfg_a/gsm8k_google_gemma-3-4b-pt_999"
    _write_prior_run(out, stale)

    with pytest.raises(FileExistsError, match="--replace"):
        _build_prior_runs(bpr, [(run, "train")], raw, out)
    assert (out / stale).is_dir()

    _build_prior_runs(bpr, [(run, "train")], raw, out, replace=True)
    assert not (out / stale).exists()
    assert (out / run / "solve_out.txt").is_file()


def test_build_prior_runs_invalid_replacement_preserves_existing_output(tmp_path: Path) -> None:
    bpr = _load(REPO / "tools" / "build_prior_runs.py")
    raw = tmp_path / "raw"
    run = "cfg_a/gsm8k_google_gemma-3-4b-pt_1"
    source = _write_prior_run(raw, run)
    out = tmp_path / "prior_runs"
    _build_prior_runs(bpr, [(run, "train")], raw, out)
    original_index = (out / "index.jsonl").read_text()
    (source / "metrics.json").unlink()

    with pytest.raises(bpr.PriorRunsError, match="metrics.json"):
        _build_prior_runs(bpr, [(run, "train")], raw, out, replace=True)
    assert (out / run / "metrics.json").is_file()
    assert (out / "index.jsonl").read_text() == original_index


def test_build_prior_runs_copy_failure_never_publishes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bpr = _load(REPO / "tools" / "build_prior_runs.py")
    raw = tmp_path / "raw"
    runs = [
        ("cfg_a/gsm8k_google_gemma-3-4b-pt_1", "train"),
        ("cfg_b/gsm8k_Qwen_Qwen3-4B-Base_2", "train"),
    ]
    for run, _side in runs:
        _write_prior_run(raw, run)
    real_copy2 = bpr.shutil.copy2
    calls = 0

    def fail_during_second_run(src, dst):
        nonlocal calls
        calls += 1
        if calls == len(bpr.MANDATORY_FILES) + 1:
            raise OSError("injected copy failure")
        return real_copy2(src, dst)

    monkeypatch.setattr(bpr.shutil, "copy2", fail_during_second_run)
    out = tmp_path / "prior_runs"
    with pytest.raises(OSError, match="injected copy failure"):
        _build_prior_runs(bpr, runs, raw, out)
    assert not out.exists()
    assert not list(tmp_path.glob(".prior_runs.tmp-*"))


def test_build_prior_runs_index_only_rejects_undeclared_or_incomplete_contents(
    tmp_path: Path,
) -> None:
    bpr = _load(REPO / "tools" / "build_prior_runs.py")
    raw = tmp_path / "raw"
    run = "cfg_a/gsm8k_google_gemma-3-4b-pt_1"
    _write_prior_run(raw, run, 0.61)
    out = tmp_path / "prior_runs"
    _build_prior_runs(bpr, [(run, "train")], raw, out)
    original_index = (out / "index.jsonl").read_text()
    _write_prior_run(out, "cfg_a/gsm8k_google_gemma-3-4b-pt_999")

    with pytest.raises(bpr.PriorRunsError, match="undeclared run"):
        _build_prior_runs(bpr, [(run, "train")], raw, out, copy=False)
    assert (out / "index.jsonl").read_text() == original_index


def test_build_prior_runs_index_only_uses_validated_copies_not_raw_source(tmp_path: Path) -> None:
    bpr = _load(REPO / "tools" / "build_prior_runs.py")
    raw = tmp_path / "raw"
    run = "cfg_a/gsm8k_google_gemma-3-4b-pt_1"
    source = _write_prior_run(raw, run, 0.61)
    out = tmp_path / "prior_runs"
    _build_prior_runs(bpr, [(run, "train")], raw, out)
    (source / "metrics.json").write_text(json.dumps({"accuracy": 0.99}))

    manifest_before = (out / "corpus-manifest.json").read_bytes()
    _build_prior_runs(bpr, [(run, "train")], raw, out, copy=False)
    row = json.loads((out / "index.jsonl").read_text())
    assert row["accuracy"] == 0.61
    assert (out / "corpus-manifest.json").read_bytes() == manifest_before

    (out / run / "time_taken.txt").unlink()
    with pytest.raises(bpr.PriorRunsError, match="time_taken.txt"):
        _build_prior_runs(bpr, [(run, "train")], raw, out, copy=False)


def test_cell_validator_attests_raw_manifest_and_detects_corruption(tmp_path: Path) -> None:
    bpr = _load(REPO / "tools" / "build_prior_runs.py")
    validator = _load(REPO / "rollout" / "validate_study_corpus.py")
    raw = tmp_path / "raw"
    run = "cfg_a/gsm8k_google_gemma-3-4b-pt_1"
    _write_prior_run(raw, run)
    out = tmp_path / "prior"
    _build_prior_runs(bpr, [(run, "train")], raw, out)
    digest = hashlib.sha256((out / "corpus-manifest.json").read_bytes()).hexdigest()

    attestation = validator.validate_raw(out, ("train",), digest)
    assert attestation["kind"] == "raw"
    assert attestation["manifest_sha256"] == digest
    assert attestation["scope"] == ["train"]
    assert attestation["run_count"] == 1
    assert set(attestation) >= {"index_sha256", "overview_sha256", "readme_sha256"}

    record = tmp_path / "study-input.json"
    invoked = subprocess.run(
        [
            sys.executable,
            str(REPO / "rollout" / "validate_study_corpus.py"),
            "raw",
            str(out),
            "--sides",
            "train",
            "--expected-manifest-sha256",
            digest,
            "--condition",
            "c1",
            "--repetition",
            "1",
            "--study-mode",
            "smoke",
            "--num-hours",
            "1",
            "--ptb-commit",
            "a" * 40,
            "--harness-commit",
            "b" * 40,
            "--ptb-surface-manifest-sha256",
            "c" * 64,
            "--record",
            str(record),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert invoked.returncode == 0, invoked.stderr
    recorded = json.loads(record.read_text())
    assert recorded["study_mode"] == "smoke" and recorded["num_hours"] == 1
    assert recorded["ptb_surface_manifest_sha256"] == "c" * 64

    (out / run / "solve_parsed.txt").write_text("unattested")
    with pytest.raises(validator.ValidationError, match="unexpected file"):
        validator.validate_raw(out, ("train",), digest)
    (out / run / "solve_parsed.txt").unlink()

    (out / run / "solve_out.txt").write_text("corrupted")
    with pytest.raises(validator.ValidationError, match="hash/size mismatch"):
        validator.validate_raw(out, ("train",), digest)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("index.jsonl", "canonical manifest-derived index"),
        ("INDEX.md", "canonical manifest-derived overview"),
        ("README.md", "canonical generated README"),
    ],
)
def test_cell_validator_rejects_mutated_raw_root_metadata(
    tmp_path: Path, name: str, message: str
) -> None:
    bpr = _load(REPO / "tools" / "build_prior_runs.py")
    validator = _load(REPO / "rollout" / "validate_study_corpus.py")
    raw = tmp_path / "raw"
    run = "cfg_a/gsm8k_google_gemma-3-4b-pt_1"
    _write_prior_run(raw, run)
    out = tmp_path / "prior"
    _build_prior_runs(bpr, [(run, "train")], raw, out)
    digest = hashlib.sha256((out / "corpus-manifest.json").read_bytes()).hexdigest()
    target = out / name
    target.write_text(target.read_text() + "\n")
    with pytest.raises(validator.ValidationError, match=message):
        validator.validate_raw(out, ("train",), digest)


def test_scientist_model_attestation_records_exact_reported_id(tmp_path: Path) -> None:
    attester = REPO / "rollout" / "attest_claude_runtime.py"
    stream = tmp_path / "scientist.jsonl"
    stream.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "type": "system",
                    "subtype": "init",
                    "model": "vertex-opus-exact",
                    "apiKeySource": "none",
                },
                {
                    "type": "assistant",
                    "message": {"model": "vertex-opus-exact", "content": []},
                },
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "modelUsage": {"vertex-opus-exact": {"provider": "vertex"}},
                },
            )
        )
        + "\n"
    )
    study = tmp_path / "study-input.json"
    study.write_text(json.dumps({"condition": "c1"}))
    record = tmp_path / "model-attestation.json"
    result = subprocess.run(
        [
            sys.executable,
            str(attester),
            "model",
            str(stream),
            "--requested-alias",
            "claude-opus-4-6",
            "--expected-model-id",
            "vertex-opus-exact",
            "--record",
            str(record),
            "--study-input",
            str(study),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    evidence = json.loads(record.read_text())
    assert evidence["requested_alias"] == "claude-opus-4-6"
    assert evidence["reported_model_ids"] == ["vertex-opus-exact"]
    assert evidence["api_key_sources"] == ["none"]
    assert evidence["reported_providers"] == ["vertex"]
    assert json.loads(study.read_text())["scientist_model"] == evidence


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"type": "result", "subtype": "success"}], "did not report"),
        (
            [
                {"type": "system", "model": "wrong-model"},
                {"type": "result", "subtype": "success"},
            ],
            "do not exactly match",
        ),
        (
            [
                {"type": "system", "model": "vertex-opus-exact"},
                {
                    "type": "result",
                    "subtype": "success",
                    "modelUsage": {"vertex-opus-exact": {}, "fallback-model": {}},
                },
            ],
            "do not exactly match",
        ),
    ],
)
def test_scientist_model_attestation_rejects_missing_wrong_or_fallback(
    tmp_path: Path, rows: list[dict], message: str
) -> None:
    attester = _load(REPO / "rollout" / "attest_claude_runtime.py")
    stream = tmp_path / "stream.jsonl"
    stream.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(attester.AttestationError, match=message):
        attester.attest_model(stream, "claude-opus-4-6", "vertex-opus-exact")


@pytest.mark.parametrize(
    ("source", "usage", "message"),
    [
        ("ANTHROPIC_API_KEY", {"provider": "vertex"}, "direct key source"),
        (None, {"provider": "vertex"}, "direct key source"),
        ("none", {}, "omitted actual provider telemetry"),
        ("none", {"provider": "firstParty"}, "not exactly Vertex"),
    ],
)
def test_scientist_model_attestation_requires_vertex_provider_telemetry(
    tmp_path: Path, source: str | None, usage: dict, message: str
) -> None:
    attester = _load(REPO / "rollout" / "attest_claude_runtime.py")
    init = {"type": "system", "subtype": "init", "model": "vertex-opus-exact"}
    if source is not None:
        init["apiKeySource"] = source
    rows = [
        init,
        {
            "type": "result",
            "subtype": "success",
            "modelUsage": {"vertex-opus-exact": usage},
        },
    ]
    stream = tmp_path / "stream.jsonl"
    stream.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(attester.AttestationError, match=message):
        attester.attest_model(stream, "claude-opus-4-6", "vertex-opus-exact")


def test_exact_claude_cli_install_is_pinned_and_attested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attester = _load(REPO / "rollout" / "attest_claude_runtime.py")
    monkeypatch.setenv("HOME", str(tmp_path))
    binary = tmp_path / ".local" / "bin" / "claude"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):
        seen.append(list(argv))
        if argv[0] == "npm":
            return subprocess.CompletedProcess(argv, 0)
        return subprocess.CompletedProcess(
            argv, 0, stdout="2.1.251 (Claude Code)\n", stderr=""
        )

    monkeypatch.setattr(attester.subprocess, "run", fake_run)
    monkeypatch.setattr(attester.shutil, "which", lambda _name: str(binary))
    version_file = tmp_path / "cli_version.txt"
    evidence = attester.install_and_attest_cli(
        version_file, "2.1.251", "2.1.251 (Claude Code)"
    )
    assert "@anthropic-ai/claude-code@2.1.251" in seen[0]
    assert evidence["package_version"] == "2.1.251"
    assert evidence["actual_version_output"] == "2.1.251 (Claude Code)"
    assert "update: pinned-install" in version_file.read_text()

    with pytest.raises(attester.AttestationError, match="does not exactly match"):
        attester.install_and_attest_cli(version_file, "2.1.251", "wrong output")


def _write_valid_wma_session(root: Path) -> tuple[Path, list[dict]]:
    session = root / "session"
    card_id = "exp-01"
    card_dir = session / "wm" / "cards" / card_id
    checkpoint = session / "checkpoints" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "gemma3"}) + "\n")
    base_checkpoint = root / "base-model"
    base_checkpoint.mkdir()
    (base_checkpoint / "config.json").write_text(json.dumps({"model_type": "gemma3"}) + "\n")

    brief_audit = card_dir / "wma-calls" / "brief-1" / "audit.json"
    observation_audit = card_dir / "wma-calls" / "observation-obs-1-1" / "audit.json"
    for audit, phase in (
        (brief_audit, "brief"),
        (observation_audit, "observation-obs-1"),
    ):
        audit.parent.mkdir(parents=True)
        audit.write_text(
            json.dumps(
                {
                    "status": "success",
                    "card_id": card_id,
                    "phase": phase,
                    "tool_event_count": 2,
                    "citation_count": 1,
                }
            )
            + "\n"
        )

    pings = card_dir / "pings"
    replies = card_dir / "replies"
    pings.mkdir()
    replies.mkdir()
    brief_ping = {
        "schema_version": "awm-ping-v1",
        "card_id": card_id,
        "ping_id": "p-1",
        "kind": "brief",
        "reply_required": True,
    }
    decision_ping = {
        "schema_version": "awm-ping-v1",
        "card_id": card_id,
        "ping_id": "p-3",
        "kind": "decision",
        "reply_required": True,
        "observation": "obs-1",
    }
    (pings / "p-1.yaml").write_text(json.dumps(brief_ping) + "\n")
    (pings / "p-3.yaml").write_text(json.dumps(decision_ping) + "\n")
    (replies / "p-1.yaml").write_text(
        json.dumps(
            {
                "card_id": card_id,
                "ping_id": "p-1",
                "choice": "accept",
                "by": "scientist",
            }
        )
        + "\n"
    )
    (replies / "p-3.yaml").write_text(
        json.dumps(
            {
                "card_id": card_id,
                "ping_id": "p-3",
                "choice": "select:obs-1",
                "by": "scientist",
            }
        )
        + "\n"
    )

    observation_dir = card_dir / "observations" / "obs-1"
    observation_dir.mkdir(parents=True)
    observation = {
        "card_id": card_id,
        "obs_id": "obs-1",
        "checkpoint": {"path": str(checkpoint), "step": 1},
        "cause": {"final": True, "standing": [1.0], "requested": []},
        "evaluators": {"dev4": {"value": 0.5}},
    }
    (observation_dir / "observation.json").write_text(json.dumps(observation) + "\n")
    seal = {
        "card_id": card_id,
        "obs_id": "obs-1",
        "checkpoint": {"path": str(checkpoint)},
        "decision_ping": "p-3",
    }
    (card_dir / "seal.json").write_text(json.dumps(seal) + "\n")
    state = {
        "card_id": card_id,
        "status": "closed",
        "final_seen": True,
        "aborted": False,
        "seal": {"obs_id": "obs-1", "checkpoint": str(checkpoint)},
    }
    (card_dir / "state.json").write_text(json.dumps(state) + "\n")
    card = {
        "schema_version": "awm-experiment-card-v1",
        "card_id": card_id,
        "setup": {
            "base_model": "google/gemma-3-4b-pt",
            "parent_checkpoint": {"path": str(base_checkpoint), "origin": "base_model"},
        },
        "result": {
            "execution": "completed",
            "output_checkpoint": str(checkpoint),
            "training_summary": {"steps": 1},
        },
        "conclusion": {"decision": "adopt"},
    }
    (card_dir / "card.yaml").write_text(json.dumps(card) + "\n")

    events = [
        {"seq": 1, "event": "session_init"},
        {"seq": 2, "event": "card_proposed", "card_id": card_id},
        {
            "seq": 3,
            "event": "wma_call",
            "card_id": card_id,
            "path": str(brief_audit),
            "phase": "brief",
            "tool_event_count": 2,
            "citation_count": 1,
        },
        {
            "seq": 4,
            "event": "ping",
            "card_id": card_id,
            "ping_id": "p-1",
            "kind": "brief",
            "reply_required": True,
            "path": str(pings / "p-1.yaml"),
        },
        {"seq": 5, "event": "reply", "card_id": card_id, "ping_id": "p-1", "choice": "accept"},
        {"seq": 6, "event": "card_frozen", "card_id": card_id},
        {"seq": 7, "event": "parent_scored", "card_id": card_id},
        {"seq": 8, "event": "training_started", "card_id": card_id},
        {"seq": 9, "event": "hook", "card_id": card_id, "step": 1, "code": 3, "final": True},
        {"seq": 10, "event": "worker_spawned", "card_id": card_id},
        {
            "seq": 11,
            "event": "observation",
            "card_id": card_id,
            "obs_id": "obs-1",
            "step": 1,
            "values": {"dev4": 0.5},
        },
        {
            "seq": 12,
            "event": "wma_call",
            "card_id": card_id,
            "path": str(observation_audit),
            "phase": "observation-obs-1",
            "tool_event_count": 2,
            "citation_count": 1,
        },
        {
            "seq": 13,
            "event": "ping",
            "card_id": card_id,
            "ping_id": "p-3",
            "kind": "decision",
            "reply_required": True,
            "path": str(pings / "p-3.yaml"),
        },
        {
            "seq": 14,
            "event": "reply",
            "card_id": card_id,
            "ping_id": "p-3",
            "choice": "select:obs-1",
        },
        {
            "seq": 15,
            "event": "decision_applied",
            "card_id": card_id,
            "ping_id": "p-3",
            "choice": "select:obs-1",
        },
        {
            "seq": 16,
            "event": "sealed",
            "card_id": card_id,
            "obs_id": "obs-1",
            "checkpoint": str(checkpoint),
        },
        {
            "seq": 17,
            "event": "awaiting_review",
            "card_id": card_id,
            "via": "select",
            "obs_id": "obs-1",
        },
        {
            "seq": 18,
            "event": "adopted",
            "card_id": card_id,
            "checkpoint": str(checkpoint),
            "submission": str(session / "final_model"),
            "mode": "copy",
        },
        {
            "seq": 19,
            "event": "card_closed",
            "card_id": card_id,
            "how": "finalize",
            "decision": "adopt",
        },
    ]
    ledger = session / "wm" / "events.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in events))
    submission = session / "final_model"
    submission.mkdir()
    (submission / "config.json").write_text("{}\n")
    (session / "wm" / "incumbent.json").write_text(
        json.dumps(
            {"card_id": card_id, "checkpoint": str(checkpoint), "obs_id": "obs-1"}
        )
        + "\n"
    )
    return session, events


def _append_valid_lineage_card(
    session: Path,
    events: list[dict],
    *,
    card_id: str = "exp-02",
    parent_card_id: str = "exp-01",
) -> tuple[list[dict], Path]:
    parent_card = json.loads(
        (session / "wm" / "cards" / parent_card_id / "card.yaml").read_text()
    )
    parent_checkpoint = parent_card["result"]["output_checkpoint"]
    checkpoint = session / "checkpoints" / f"checkpoint-{card_id}"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "gemma3"}) + "\n")

    card_dir = session / "wm" / "cards" / card_id
    audit = card_dir / "wma-calls" / "brief-1" / "audit.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(
        json.dumps(
            {
                "status": "success",
                "card_id": card_id,
                "phase": "brief",
                "tool_event_count": 1,
                "citation_count": 1,
            }
        )
        + "\n"
    )
    card = {
        "schema_version": "awm-experiment-card-v1",
        "card_id": card_id,
        "setup": {
            "base_model": "google/gemma-3-4b-pt",
            "parent_checkpoint": {
                "path": parent_checkpoint,
                "origin": parent_card_id,
            },
        },
        "result": {
            "execution": "completed",
            "output_checkpoint": str(checkpoint),
            "training_summary": {"steps": 1},
        },
        "conclusion": {"decision": "adopt"},
    }
    (card_dir / "card.yaml").write_text(json.dumps(card) + "\n")

    appended = [
        {"event": "card_proposed", "card_id": card_id},
        {
            "event": "wma_call",
            "card_id": card_id,
            "path": str(audit),
            "phase": "brief",
            "tool_event_count": 1,
            "citation_count": 1,
        },
        {"event": "training_started", "card_id": card_id},
        {
            "event": "sealed",
            "card_id": card_id,
            "obs_id": "obs-2",
            "checkpoint": str(checkpoint),
        },
        {
            "event": "adopted",
            "card_id": card_id,
            "checkpoint": str(checkpoint),
            "submission": str(session / "final_model"),
            "mode": "copy",
        },
        {
            "event": "card_closed",
            "card_id": card_id,
            "how": "finalize",
            "decision": "adopt",
        },
    ]
    next_sequence = max(row["seq"] for row in events) + 1
    for sequence, row in enumerate(appended, next_sequence):
        row["seq"] = sequence
    events = [*events, *appended]
    (session / "wm" / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    (session / "wm" / "incumbent.json").write_text(
        json.dumps(
            {"card_id": card_id, "checkpoint": str(checkpoint), "obs_id": "obs-2"}
        )
        + "\n"
    )
    return events, checkpoint


def test_wma_session_postcondition_requires_successful_call_and_adoption(tmp_path: Path) -> None:
    validator = _load(REPO / "rollout" / "validate_wma_session.py")
    session, _events = _write_valid_wma_session(tmp_path)
    evidence = validator.validate(session)
    assert evidence["adopted_card_ids"] == ["exp-01"]
    assert evidence["successful_wma_call_count"] == 2


@pytest.mark.parametrize(
    ("remove_event", "append_event", "message"),
    [
        ("wma_call", None, "no successful wma_call"),
        ("sealed", None, "sealed, adopted, and finalized"),
        (None, {"seq": 5, "event": "agent_degraded"}, "agent_failed or agent_degraded"),
    ],
)
def test_wma_session_postcondition_rejects_incomplete_or_degraded(
    tmp_path: Path,
    remove_event: str | None,
    append_event: dict | None,
    message: str,
) -> None:
    validator = _load(REPO / "rollout" / "validate_wma_session.py")
    session, events = _write_valid_wma_session(tmp_path)
    if remove_event:
        events = [row for row in events if row["event"] != remove_event]
        for seq, row in enumerate(events, 1):
            row["seq"] = seq
    if append_event:
        append_event["seq"] = max(row["seq"] for row in events) + 1
        events.append(append_event)
    (session / "wm" / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    with pytest.raises(validator.ValidationError, match=message):
        validator.validate(session)


def test_wma_smoke_postcondition_attests_correlated_full_lifecycle(tmp_path: Path) -> None:
    validator = _load(REPO / "rollout" / "validate_wma_session.py")
    session, _events = _write_valid_wma_session(tmp_path)
    evidence = validator.validate(
        session,
        require_smoke_lifecycle=True,
        expected_base_model="google/gemma-3-4b-pt",
        expected_base_checkpoint=tmp_path / "base-model",
    )
    lifecycle = evidence["smoke_lifecycle"]
    assert lifecycle["card_id"] == "exp-01"
    assert lifecycle["brief_ping_id"] == "p-1"
    assert lifecycle["decision_ping_id"] == "p-3"
    assert lifecycle["observation_id"] == "obs-1"
    assert lifecycle["training_steps"] == 1
    assert lifecycle["base_model"] == "google/gemma-3-4b-pt"


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("missing_proposal", "missing card_proposed"),
        ("brief_timeout", "missing explicit scientist brief reply"),
        ("hook_not_final", "missing final checkpoint hook yield"),
        ("observation_step", "missing observation for the final hook"),
        ("observation_phase", "missing successful observation wma_call"),
        ("decision_observation", "decision ping does not name"),
        ("decision_timeout", "missing explicit scientist decision reply"),
        ("wrong_selection", "missing explicit scientist decision reply"),
        ("wrong_seal", "missing seal for the selected observation"),
        ("unsloth_base", "base model is not google/gemma-3-4b-pt"),
        ("wrong_base_checkpoint", "parent is not the expected official base checkpoint"),
        ("zero_training_steps", "finalized card is incomplete"),
    ],
)
def test_wma_smoke_postcondition_rejects_uncorrelated_or_wrong_base(
    tmp_path: Path, damage: str, message: str
) -> None:
    validator = _load(REPO / "rollout" / "validate_wma_session.py")
    session, events = _write_valid_wma_session(tmp_path)
    card_dir = session / "wm" / "cards" / "exp-01"

    if damage == "missing_proposal":
        events = [row for row in events if row["event"] != "card_proposed"]
    elif damage == "brief_timeout":
        row = next(row for row in events if row["event"] == "reply" and row["ping_id"] == "p-1")
        row["event"] = "timeout"
        reply = json.loads((card_dir / "replies" / "p-1.yaml").read_text())
        reply["by"] = "timeout"
        (card_dir / "replies" / "p-1.yaml").write_text(json.dumps(reply) + "\n")
    elif damage == "hook_not_final":
        next(row for row in events if row["event"] == "hook")["final"] = False
    elif damage == "observation_step":
        next(row for row in events if row["event"] == "observation")["step"] = 2
    elif damage == "observation_phase":
        next(
            row
            for row in events
            if row["event"] == "wma_call" and row.get("phase", "").startswith("observation-")
        )["phase"] = "observation-obs-2"
    elif damage == "decision_observation":
        ping = json.loads((card_dir / "pings" / "p-3.yaml").read_text())
        ping["observation"] = "obs-2"
        (card_dir / "pings" / "p-3.yaml").write_text(json.dumps(ping) + "\n")
    elif damage == "decision_timeout":
        row = next(row for row in events if row["event"] == "reply" and row["ping_id"] == "p-3")
        row["event"] = "timeout"
        reply = json.loads((card_dir / "replies" / "p-3.yaml").read_text())
        reply["by"] = "timeout"
        (card_dir / "replies" / "p-3.yaml").write_text(json.dumps(reply) + "\n")
    elif damage == "wrong_selection":
        row = next(row for row in events if row["event"] == "reply" and row["ping_id"] == "p-3")
        row["choice"] = "select:obs-2"
        reply = json.loads((card_dir / "replies" / "p-3.yaml").read_text())
        reply["choice"] = "select:obs-2"
        (card_dir / "replies" / "p-3.yaml").write_text(json.dumps(reply) + "\n")
    elif damage == "wrong_seal":
        next(row for row in events if row["event"] == "sealed")["checkpoint"] = "other"
    elif damage == "unsloth_base":
        card = json.loads((card_dir / "card.yaml").read_text())
        card["setup"]["base_model"] = "unsloth/gemma-3-4b-pt"
        (card_dir / "card.yaml").write_text(json.dumps(card) + "\n")
    elif damage == "wrong_base_checkpoint":
        wrong = tmp_path / "other-base"
        wrong.mkdir()
        card = json.loads((card_dir / "card.yaml").read_text())
        card["setup"]["parent_checkpoint"]["path"] = str(wrong)
        (card_dir / "card.yaml").write_text(json.dumps(card) + "\n")
    elif damage == "zero_training_steps":
        card = json.loads((card_dir / "card.yaml").read_text())
        card["result"]["training_summary"]["steps"] = 0
        (card_dir / "card.yaml").write_text(json.dumps(card) + "\n")

    (session / "wm" / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    with pytest.raises(validator.ValidationError, match=message):
        validator.validate(
            session,
            require_smoke_lifecycle=True,
            expected_base_model="google/gemma-3-4b-pt",
            expected_base_checkpoint=tmp_path / "base-model",
        )


def test_wma_production_postcondition_keeps_broad_existing_semantics(tmp_path: Path) -> None:
    validator = _load(REPO / "rollout" / "validate_wma_session.py")
    session, events = _write_valid_wma_session(tmp_path)
    events = [
        next(row for row in events if row["event"] == "wma_call"),
        next(row for row in events if row["event"] == "training_started"),
        next(row for row in events if row["event"] == "sealed"),
        next(row for row in events if row["event"] == "adopted"),
        next(row for row in events if row["event"] == "card_closed"),
    ]
    for seq, row in enumerate(events, 1):
        row["seq"] = seq
    (session / "wm" / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    study_input = session / "study-input.json"
    study_input.write_text(json.dumps({"study_mode": "production"}) + "\n")
    record = session / "wma-session-attestation.json"
    assert validator.main(
        [
            str(session),
            "--record",
            str(record),
            "--study-input",
            str(study_input),
            "--expected-base-model",
            "google/gemma-3-4b-pt",
            "--expected-base-checkpoint",
            str(tmp_path / "base-model"),
        ]
    ) == 0
    evidence = json.loads(record.read_text())
    assert evidence["adopted_card_ids"] == ["exp-01"]
    assert "smoke_lifecycle" not in evidence
    assert evidence["base_lineage"]["final_card_id"] == "exp-01"


def test_wma_production_base_lineage_accepts_ordered_two_card_chain(
    tmp_path: Path,
) -> None:
    validator = _load(REPO / "rollout" / "validate_wma_session.py")
    session, events = _write_valid_wma_session(tmp_path)
    _events, checkpoint = _append_valid_lineage_card(session, events)
    evidence = validator.validate(
        session,
        expected_base_model="google/gemma-3-4b-pt",
        expected_base_checkpoint=tmp_path / "base-model",
    )
    lineage = evidence["base_lineage"]
    assert lineage["executed_card_ids"] == ["exp-01", "exp-02"]
    assert lineage["final_card_id"] == "exp-02"
    assert lineage["final_checkpoint"] == str(checkpoint)
    assert [card["parent"] for card in lineage["cards"]] == [
        "base_model",
        "exp-01",
    ]


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("wrong_base_model", "base model is not google/gemma-3-4b-pt"),
        ("wrong_root_checkpoint", "does not reference the official base"),
        ("unknown_parent", "does not reference the official base or one executed card"),
        ("wrong_parent_origin", "parent origin does not name exp-01"),
        ("forward_parent", "does not follow an earlier finalized card output"),
        ("cycle", "contains a cycle"),
        ("missing_parent_finalization", "has no terminal card_closed/finalize"),
        ("missing_output", "output checkpoint is missing or linked"),
        ("final_without_training", "has no training_started event"),
        ("incumbent_mismatch", "does not match the final adopted event"),
    ],
)
def test_wma_production_base_lineage_rejects_invalid_chains(
    tmp_path: Path, damage: str, message: str
) -> None:
    validator = _load(REPO / "rollout" / "validate_wma_session.py")
    session, events = _write_valid_wma_session(tmp_path)
    events, second_checkpoint = _append_valid_lineage_card(session, events)
    first_path = session / "wm" / "cards" / "exp-01" / "card.yaml"
    second_path = session / "wm" / "cards" / "exp-02" / "card.yaml"
    first = json.loads(first_path.read_text())
    second = json.loads(second_path.read_text())

    if damage == "wrong_base_model":
        second["setup"]["base_model"] = "unsloth/gemma-3-4b-pt"
    elif damage == "wrong_root_checkpoint":
        wrong_root = tmp_path / "other-base-model"
        wrong_root.mkdir()
        first["setup"]["parent_checkpoint"]["path"] = str(wrong_root)
    elif damage == "unknown_parent":
        unknown = session / "checkpoints" / "unknown-parent"
        unknown.mkdir()
        second["setup"]["parent_checkpoint"]["path"] = str(unknown)
    elif damage == "wrong_parent_origin":
        second["setup"]["parent_checkpoint"]["origin"] = "incumbent"
    elif damage == "forward_parent":
        first["setup"]["parent_checkpoint"] = {
            "path": str(second_checkpoint),
            "origin": "exp-02",
        }
        second["setup"]["parent_checkpoint"] = {
            "path": str(tmp_path / "base-model"),
            "origin": "base_model",
        }
    elif damage == "cycle":
        first["setup"]["parent_checkpoint"] = {
            "path": str(second_checkpoint),
            "origin": "exp-02",
        }
    elif damage == "missing_parent_finalization":
        events = [
            row
            for row in events
            if not (row["event"] == "card_closed" and row.get("card_id") == "exp-01")
        ]
        for sequence, row in enumerate(events, 1):
            row["seq"] = sequence
    elif damage == "missing_output":
        second["result"]["output_checkpoint"] = str(
            session / "checkpoints" / "missing-output"
        )
    elif damage == "final_without_training":
        events = [
            row
            for row in events
            if not (
                row["event"] == "training_started" and row.get("card_id") == "exp-02"
            )
        ]
        for sequence, row in enumerate(events, 1):
            row["seq"] = sequence
    elif damage == "incumbent_mismatch":
        (session / "wm" / "incumbent.json").write_text(
            json.dumps(
                {
                    "card_id": "exp-01",
                    "checkpoint": first["result"]["output_checkpoint"],
                    "obs_id": "obs-1",
                }
            )
            + "\n"
        )

    first_path.write_text(json.dumps(first) + "\n")
    second_path.write_text(json.dumps(second) + "\n")
    (session / "wm" / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    with pytest.raises(validator.ValidationError, match=message):
        validator.validate(
            session,
            expected_base_model="google/gemma-3-4b-pt",
            expected_base_checkpoint=tmp_path / "base-model",
        )


def test_wma_cli_enforces_correlated_lifecycle_only_for_smoke(tmp_path: Path) -> None:
    validator = _load(REPO / "rollout" / "validate_wma_session.py")
    session, _events = _write_valid_wma_session(tmp_path)
    study_input = session / "study-input.json"
    study_input.write_text(json.dumps({"study_mode": "smoke"}) + "\n")
    record = session / "wma-session-attestation.json"
    assert validator.main(
        [
            str(session),
            "--record",
            str(record),
            "--study-input",
            str(study_input),
            "--expected-base-model",
            "google/gemma-3-4b-pt",
            "--expected-base-checkpoint",
            str(tmp_path / "base-model"),
        ]
    ) == 0
    evidence = json.loads(record.read_text())
    assert evidence["smoke_lifecycle"]["card_id"] == "exp-01"
    assert json.loads(study_input.read_text())["wma_session"] == evidence


def test_cell_validator_attests_exact_card_scope(tmp_path: Path) -> None:
    validator = _load(REPO / "rollout" / "validate_study_corpus.py")
    root = tmp_path / "memory"
    side_root = root / "corpus" / "train"
    card = side_root / "r-1" / "exp-0001.yaml"
    card.parent.mkdir(parents=True)
    card.write_text("card_id: exp-0001\n")
    card_digest = hashlib.sha256(card.read_bytes()).hexdigest()
    (side_root / "manifest.jsonl").write_text(
        json.dumps(
            {
                "card_id": "exp-0001",
                "path": "r-1/exp-0001.yaml",
                "sha256": card_digest,
                "run_ref": "r-1",
                "side": "train",
            }
        )
        + "\n"
    )
    (side_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "awm-exp-card-corpus-v1",
                "side": "train",
                "card_count": 1,
                "card_bearing_run_count": 1,
                "expected_run_count": 1,
                "missing_run_count": 0,
            },
            sort_keys=True,
        )
        + "\n"
    )
    structured = root / "structured"
    structured.mkdir()
    (structured / "cards.jsonl").write_text(
        json.dumps({"provenance": {"split_side": "train"}}) + "\n"
    )

    attestation = validator.validate_cards(root, ("train",))
    assert attestation["kind"] == "cards"
    assert attestation["scope"] == ["train"]
    assert attestation["card_count"] == 1
    assert len(attestation["manifest_sha256"]) == 64

    rogue = side_root / "r-1" / "exp-rogue.yml"
    rogue.write_text("unattested: true\n")
    with pytest.raises(validator.ValidationError, match="filesystem inventory mismatch"):
        validator.validate_cards(root, ("train",))
    rogue.unlink()

    (root / "corpus" / "test").mkdir()
    with pytest.raises(validator.ValidationError, match="do not exactly match"):
        validator.validate_cards(root, ("train",))

def test_build_prior_runs_index_only_validates_manifest_hashes_and_revision(tmp_path: Path) -> None:
    bpr = _load(REPO / "tools" / "build_prior_runs.py")
    raw = tmp_path / "raw"
    run = "cfg_a/gsm8k_google_gemma-3-4b-pt_1"
    _write_prior_run(raw, run)
    out = tmp_path / "prior_runs"
    _build_prior_runs(bpr, [(run, "train")], raw, out)
    (out / run / "solve_out.txt").write_text("tampered trajectory")

    with pytest.raises(bpr.PriorRunsError, match="current file hashes"):
        _build_prior_runs(bpr, [(run, "train")], raw, out, copy=False)

    # Restore the copied bytes, then prove provenance is independently part of
    # the immutable comparison.
    (out / run / "solve_out.txt").write_text("trace " * 100)
    wrong_dataset = {**TEST_PRIOR_DATASET, "revision": "a" * 40}
    with pytest.raises(bpr.PriorRunsError, match="split/revision/sides"):
        _build_prior_runs(
            bpr,
            [(run, "train")],
            raw,
            out,
            copy=False,
            dataset=wrong_dataset,
        )


def test_build_prompts_renders_ptb_placeholders() -> None:
    bp = _load(REPO / "rollout" / "build_prompts.py")
    ptb = "Train `{model}` on {benchmark} for {num_hours} hours.\n## Rules\n1. Keep going.\n"
    wm = bp.wm_prompt(ptb, fulltraj=False)
    assert "{model}" in wm and "{benchmark}" in wm and "{num_hours} hours" in wm
    assert "## Pinned base checkpoint" in wm
    assert "## The world-model agent" in wm
    assert "SendMessage" in wm
    assert "## Prior runs" not in wm
    wm_ft = bp.wm_prompt(ptb, fulltraj=True)
    assert wm_ft.index("## Prior runs") < wm_ft.index("## The world-model agent")
    prior = bp.PRIOR_RUNS_SECTION
    for guaranteed in ("`solve_out.txt`", "`metrics.json`", "`time_taken.txt`"):
        assert guaranteed in prior
    assert "Every run directory has exactly" in prior
    assert "Optional upstream artifacts and `task/` workspace snapshots" in prior
    assert "solve_parsed.txt" not in prior
    for name in (
        "prompt_fulltraj.txt",
        "prompt_wm_fulltraj.txt",
        "prompt_wm_fulltraj_smoke.txt",
    ):
        assert prior in (REPO / "rollout" / "prompts" / name).read_text()
    assert prior not in (REPO / "rollout" / "prompts" / "prompt_wm.txt").read_text()
    smoke = (REPO / "rollout" / "prompts" / "prompt_wm_fulltraj_smoke.txt").read_text()
    production = (REPO / "rollout" / "prompts" / "prompt_wm_fulltraj.txt").read_text()
    assert "## One-hour peer-session smoke protocol" in smoke
    assert "## One-hour peer-session smoke protocol" not in production
    assert "message the world-model agent" in smoke
    assert "tell the world-model agent what you shipped" in smoke
    assert "one optimizer step" in smoke
    assert "/home/ben/pinned-base/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d" in smoke
    assert "unsloth/*" in smoke
    ptb = "intro {model}\n## Rules\n1. x\n"
    assert bp.ptb_fulltraj(ptb).index("## Prior runs") < bp.ptb_fulltraj(ptb).index("## Rules")
    with pytest.raises(SystemExit):
        bp.ptb_fulltraj("no rules heading")


def test_extra_binds_patch_is_idempotent(tmp_path: Path) -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_extra_binds.py")
    original = _pinned_ptb_run_task()
    once = patcher.apply(original)
    assert once != original
    assert 'EXTRA_BIND_ARGS+=(--bind "$_b")' in once
    # exactly one exec line gains the extra binds, right after the HF cache bind
    agent_block = once.split(patcher.MARK, 1)[1]
    assert agent_block.count('"${EXTRA_BIND_ARGS[@]}" \\') == 1
    assert once.count('"${EXTRA_BIND_ARGS[@]}" \\') == 1
    assert patcher.apply(once) == once


def test_scratch_root_patch_is_idempotent_and_removes_broad_cleanup(
    tmp_path: Path,
) -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_scratch_root.py")
    original = _pinned_ptb_run_task()
    once = patcher.apply(original)
    assert once != original
    assert patcher.apply(once) == once
    assert 'POST_TRAIN_BENCH_TMP_ROOT:?set POST_TRAIN_BENCH_TMP_ROOT' in once
    assert 'mktemp -d "${AWM_SCRATCH_ROOT}/posttrain_container_' in once
    assert 'posttrain_container_${EVALUATION_TASK}' not in once
    assert "trap awm_exit_with_scratch_cleanup EXIT" in once
    assert "rm -rf /tmp/posttrain_container" not in once
    candidate = tmp_path / "run_task.sh"
    candidate.write_text(once)
    subprocess.run(["bash", "-n", str(candidate)], check=True)


def test_scratch_root_block_creates_and_cleans_only_private_cell_dir(
    tmp_path: Path,
) -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_scratch_root.py")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    sentinel = scratch / "keep"
    sentinel.write_text("safe")
    similarly_named_sibling = scratch / "posttrain_container_other-cell"
    similarly_named_sibling.mkdir()
    harness = (
        "set -euo pipefail\n"
        "EVALUATION_TASK=gsm8k\n"
        "RESULT_PREFIX_SAFE=google_gemma-3-4b-pt\n"
        "RANDOM_UUID=unit-test\n"
        + patcher.SETUP_BLOCK
        + "test -d \"${TMP_SUBDIR}\"\n"
        + "printf '%s\\n' \"${TMP_SUBDIR}\"\n"
    )
    result = subprocess.run(
        ["bash", "-s"],
        input=harness,
        env={
            "PATH": os.environ["PATH"],
            "POST_TRAIN_BENCH_TMP_ROOT": str(scratch),
            "POST_TRAIN_BENCH_MIN_SCRATCH_FREE_BYTES": "1",
            "POST_TRAIN_BENCH_MIN_SCRATCH_FREE_INODES": "1",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    created = Path(result.stdout.strip())
    assert created.parent == scratch
    assert not created.exists()
    assert sentinel.read_text() == "safe"
    assert similarly_named_sibling.is_dir()


def test_scratch_root_cleanup_failure_is_fatal_without_errexit(tmp_path: Path) -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_scratch_root.py")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_rm = fake_bin / "rm"
    fake_rm.write_text("#!/bin/bash\nexit 1\n")
    fake_rm.chmod(0o700)
    # The pinned PTB runner intentionally does not set `set -e`; the cleanup
    # function itself must therefore preserve and report an rm failure.
    harness = (
        "EVALUATION_TASK=gsm8k\n"
        "RESULT_PREFIX_SAFE=model\n"
        "RANDOM_UUID=test\n"
        + patcher.SETUP_BLOCK
        + "printf 'body-complete\\n'\n"
    )
    result = subprocess.run(
        ["bash", "-s"],
        input=harness,
        env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "POST_TRAIN_BENCH_TMP_ROOT": str(scratch),
            "POST_TRAIN_BENCH_MIN_SCRATCH_FREE_BYTES": "1",
            "POST_TRAIN_BENCH_MIN_SCRATCH_FREE_INODES": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "could not remove owned cell scratch directory" in result.stderr
    assert len(list(scratch.glob("posttrain_container_*"))) == 1


def test_scratch_root_cleanup_preserves_existing_failure(tmp_path: Path) -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_scratch_root.py")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    harness = (
        "EVALUATION_TASK=gsm8k\n"
        "RESULT_PREFIX_SAFE=model\n"
        "RANDOM_UUID=test\n"
        + patcher.SETUP_BLOCK
        + "exit 17\n"
    )
    result = subprocess.run(
        ["bash", "-s"],
        input=harness,
        env={
            "PATH": os.environ["PATH"],
            "POST_TRAIN_BENCH_TMP_ROOT": str(scratch),
            "POST_TRAIN_BENCH_MIN_SCRATCH_FREE_BYTES": "1",
            "POST_TRAIN_BENCH_MIN_SCRATCH_FREE_INODES": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 17
    assert list(scratch.iterdir()) == []


def test_scratch_root_block_rejects_insufficient_headroom(tmp_path: Path) -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_scratch_root.py")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    harness = (
        "set -euo pipefail\n"
        "EVALUATION_TASK=gsm8k\nRESULT_PREFIX_SAFE=model\nRANDOM_UUID=test\n"
        + patcher.SETUP_BLOCK
    )
    result = subprocess.run(
        ["bash", "-s"],
        input=harness,
        env={
            "PATH": os.environ["PATH"],
            "POST_TRAIN_BENCH_TMP_ROOT": str(scratch),
            "POST_TRAIN_BENCH_MIN_SCRATCH_FREE_BYTES": str(2**63 - 1),
            "POST_TRAIN_BENCH_MIN_SCRATCH_FREE_INODES": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "lacks required free blocks/inodes" in result.stderr
    assert list(scratch.iterdir()) == []


def test_scratch_root_block_rejects_exposed_or_aliased_root(tmp_path: Path) -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_scratch_root.py")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o755)
    scratch.chmod(0o755)
    harness = (
        "set -euo pipefail\n"
        "EVALUATION_TASK=gsm8k\nRESULT_PREFIX_SAFE=model\nRANDOM_UUID=test\n"
        + patcher.SETUP_BLOCK
    )
    common_env = {
        "PATH": os.environ["PATH"],
        "POST_TRAIN_BENCH_MIN_SCRATCH_FREE_BYTES": "1",
        "POST_TRAIN_BENCH_MIN_SCRATCH_FREE_INODES": "1",
    }
    exposed = subprocess.run(
        ["bash", "-s"],
        input=harness,
        env={**common_env, "POST_TRAIN_BENCH_TMP_ROOT": str(scratch)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert exposed.returncode == 2
    assert "owned by this uid and mode 0700" in exposed.stderr

    scratch.chmod(0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(scratch, target_is_directory=True)
    linked = subprocess.run(
        ["bash", "-s"],
        input=harness,
        env={**common_env, "POST_TRAIN_BENCH_TMP_ROOT": str(alias)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert linked.returncode == 2
    assert "must be a real directory" in linked.stderr


def _pinned_ptb_run_task() -> str:
    configured = os.environ.get("PTB_SOURCE_DIR")
    ptb = Path(configured) if configured else REPO / "third_party" / "PostTrainBench"
    if not (ptb / ".git").exists():
        pytest.skip("PostTrainBench submodule not checked out")
    result = subprocess.run(
        ["git", "-C", str(ptb), "show", "HEAD:src/run_task.sh"],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_study_runner_patch_applies_to_pinned_head_and_is_idempotent(tmp_path: Path) -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_study_runner.py")
    extra = _load(REPO / "rollout" / "patches" / "apply_extra_binds.py")
    scratch = _load(REPO / "rollout" / "patches" / "apply_scratch_root.py")
    original = _pinned_ptb_run_task()
    once = patcher.apply(original)
    assert once != original
    assert patcher.apply(once) == once

    combined = scratch.apply(extra.apply(once))
    assert extra.apply(combined) == combined
    assert scratch.apply(combined) == combined
    candidate = tmp_path / "run_task.sh"
    candidate.write_text(combined)
    subprocess.run(["bash", "-n", str(candidate)], check=True)

    assert 'agents/${AGENT}/payload/." "${JOB_DIR}/agent/' in combined
    assert 'STUDY_PROMPT_FILE="${JOB_DIR}/task/instruction.md"' in combined
    assert "sha256sum instruction.md > instruction.sha256" in combined
    assert '"${PROMPT_ENV_ARGS[@]}" \\' in combined
    assert '"${PROMPT_BIND_ARGS[@]}" \\' in combined
    assert "/home/ben/task/instruction.md:ro" in combined
    assert "prompt generation failed" in combined
    assert patcher.PROMPT_ARG_ANCHOR not in combined
    assert '"${AGENT_ENV_ARGS[@]}" \\' in combined
    assert 'bash -o pipefail -c "{ echo OS-visible GPU isolation probe' in combined
    assert "SOLVE_RC=\\$?" in combined
    assert 'solve_exit_code.txt' in combined
    assert 'exit "$SOLVE_EXIT"' in combined
    assert "skipping optional judges and evaluating the preserved final_model" in combined
    assert 'POST_TRAIN_BENCH_VISIBLE_GPUS' in combined
    assert 'POST_TRAIN_BENCH_ISOLATE_GPUS' in combined
    assert 'POST_TRAIN_BENCH_EVAL_GPU_REAP' in combined
    assert 'POST_TRAIN_BENCH_TMP_ROOT' in combined
    assert 'trap awm_exit_with_scratch_cleanup EXIT' in combined
    assert 'rm -rf /tmp/posttrain_container' not in combined
    assert 'POST_TRAIN_BENCH_CELL_TOKEN' in combined
    assert "OS-visible GPU isolation probe" in combined
    assert "env -u CUDA_VISIBLE_DEVICES -u NVIDIA_VISIBLE_DEVICES" in combined
    # A tuple is intentionally not valid JSON: PTB's Claude parser must not
    # mistake the diagnostic device count for a JSON event object.
    assert "print((count,))" in combined
    assert "print(count)" not in combined
    assert "count == 1 else 86" in combined
    assert "math.isfinite(float(accuracy))" in combined
    assert 'nvidia-smi --query-compute-apps=pid --format=csv,noheader |' not in combined
    assert 'xargs -r kill -9' not in combined


def test_study_runner_patch_rejects_partial_or_changed_runner() -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_study_runner.py")
    original = _pinned_ptb_run_task()
    partial = original + "\n" + patcher.MARK + "\n"
    with pytest.raises(SystemExit, match="patch is incomplete"):
        patcher.apply(partial)

    changed = original.replace("GPU_PIN_ENV=()", "RENAMED_GPU_PIN_ENV=()", 1)
    with pytest.raises(SystemExit, match="GPU pin anchor"):
        patcher.apply(changed)

    current = patcher.apply(original)
    damaged = current.replace(patcher.PROMPT_BIND_LINE, "", 1)
    with pytest.raises(SystemExit, match="prompt handoff is incomplete"):
        patcher.apply(damaged)


def test_study_runner_patch_upgrades_known_earlier_revision() -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_study_runner.py")
    current = patcher.apply(_pinned_ptb_run_task())
    older = current.replace(patcher.PROMPT_LOAD_REPLACEMENT, patcher.PROMPT_LOAD_ANCHOR)
    older = older.replace(patcher.PROMPT_ANCHOR + patcher.PROMPT_BLOCK, patcher.PROMPT_ANCHOR)
    older = older.replace(patcher.PROMPT_ARG_LINE, patcher.PROMPT_ARG_ANCHOR)
    older = older.replace(patcher.PROMPT_BIND_LINE, "")
    older = older.replace(patcher.NEW_SOLVE_LINE, patcher.NEW_SOLVE_LINE_V2)
    older = older.replace(patcher.EOF_REPLACEMENT, patcher.EOF_REPLACEMENT_V2)
    assert patcher.apply(older) == current


@pytest.mark.parametrize(("command", "expected"), (("false", 2), ("true", 2)))
def test_study_runner_rejects_failed_or_blank_prompt_generation(
    command: str, expected: int
) -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_study_runner.py")
    block = patcher.PROMPT_LOAD_REPLACEMENT.replace(patcher.PROMPT_COMMAND, command)
    result = subprocess.run(
        ["bash", "-s"],
        input="set -uo pipefail\n" + block + "printf 'unexpected=<%s>\\n' \"$PROMPT\"\n",
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == expected
    assert "ERROR: prompt generation" in result.stderr


@pytest.mark.parametrize("agent", ("claude_wm", "hv_noop"))
def test_study_runner_prompt_handoff_preserves_exact_multiline_bytes(
    tmp_path: Path, agent: str
) -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_study_runner.py")
    eval_dir = tmp_path / "results"
    job_dir = tmp_path / "job"
    eval_dir.mkdir()
    (job_dir / "task").mkdir(parents=True)
    prompt = "## Parameters\n\n- internal: a=b\n- shell data: $(touch never) * ' \"\n"
    result = subprocess.run(
        ["bash", "-s"],
        input=(
            "set -euo pipefail\n"
            + patcher.PROMPT_ANCHOR
            + patcher.PROMPT_BLOCK
            + "printf 'ARG=<%s>\\n' \"${PROMPT_ENV_ARGS[@]}\"\n"
        ),
        cwd=tmp_path,
        env={
            **os.environ,
            "AGENT": agent,
            "EVAL_DIR": str(eval_dir),
            "JOB_DIR": str(job_dir),
            "PROMPT": prompt,
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    recorded = (eval_dir / "prompt.txt").read_bytes()
    instruction = (job_dir / "task" / "instruction.md").read_bytes()
    assert recorded == instruction == (prompt + "\n").encode()
    subprocess.run(
        ["sha256sum", "--strict", "--check", "instruction.sha256"],
        cwd=job_dir / "task",
        check=True,
        capture_output=True,
    )
    if agent == "claude_wm":
        assert "ARG=<STUDY_PROMPT_SHA256=" in result.stdout
        assert "ARG=<STUDY_PROMPT_BYTES=" in result.stdout
        assert "ARG=<PROMPT=" not in result.stdout
    else:
        assert "ARG=<PROMPT=## Parameters" in result.stdout
        assert "ARG=<STUDY_PROMPT_SHA256=" not in result.stdout
    assert not (tmp_path / "never").exists()


def _study_prompt_verifier_block(solve: str) -> str:
    start = solve.index("STUDY_PROMPT_FILE=/home/ben/task/instruction.md")
    end = solve.index("\nverify_study_prompt || {", start)
    return solve[start:end]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (("none", 0), ("truncate", 1), ("checksum", 1), ("symlink", 1)),
)
def test_study_prompt_verifier_fails_closed(
    tmp_path: Path, mutation: str, expected: int
) -> None:
    agent_root = REPO / "rollout" / "agents"
    solves = [
        (agent_root / agent / "solve.sh").read_text()
        for agent in ("claude_fulltraj_noawm", "claude_wm")
    ]
    blocks = [_study_prompt_verifier_block(solve) for solve in solves]
    assert blocks[0] == blocks[1]

    task = tmp_path / "task"
    task.mkdir()
    prompt = "## Parameters\n\n- internal: a=b\n- Unicode: λ\n\n".encode()
    expected_sha = hashlib.sha256(prompt).hexdigest()
    instruction = task / "instruction.md"
    if mutation == "symlink":
        (task / "prompt-target").write_bytes(prompt)
        instruction.symlink_to("prompt-target")
    else:
        instruction.write_bytes(prompt[:-1] if mutation == "truncate" else prompt)
    checksum = expected_sha if mutation != "checksum" else "0" * 64
    (task / "instruction.sha256").write_text(f"{checksum}  instruction.md\n")

    block = blocks[0].replace("/home/ben/task", str(task))
    result = subprocess.run(
        ["bash", "-s"],
        input="set -uo pipefail\n" + block + "\nverify_study_prompt\n",
        env={
            **os.environ,
            "STUDY_PROMPT_SHA256": expected_sha,
            "STUDY_PROMPT_BYTES": str(len(prompt)),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == expected


def test_study_agent_streams_prompt_file_byte_exactly_to_claude(tmp_path: Path) -> None:
    solve = (REPO / "rollout" / "agents" / "claude_wm" / "solve.sh").read_text()
    verifier = _study_prompt_verifier_block(solve)
    pipeline_start = solve.index('cat "${STUDY_PROMPT_FILE}" | claude --print')
    pipeline_end = solve.index('\npipeline_status=("${PIPESTATUS[@]}")', pipeline_start)
    pipeline = solve[pipeline_start:pipeline_end]

    task = tmp_path / "task"
    fake_bin = tmp_path / "bin"
    task.mkdir()
    fake_bin.mkdir()
    prompt = "## Parameters\n\n- internal: a=b\n- shell: $(false) * ' \"\n- Unicode: λ\n\n".encode()
    expected_sha = hashlib.sha256(prompt).hexdigest()
    (task / "instruction.md").write_bytes(prompt)
    (task / "instruction.sha256").write_text(f"{expected_sha}  instruction.md\n")
    fake_claude = fake_bin / "claude"
    fake_claude.write_text("#!/bin/bash\nset -euo pipefail\ntee \"$CAPTURE\"\n")
    fake_claude.chmod(0o755)
    capture = tmp_path / "claude-stdin.bin"
    stream = tmp_path / "scientist-stream.bin"

    script = (
        "set -uo pipefail\n"
        + verifier.replace("/home/ben/task", str(task))
        + "\nverify_study_prompt\n"
        + pipeline
        + "\npipeline_status=(\"${PIPESTATUS[@]}\")\n"
        + "[ \"${pipeline_status[*]}\" = \"0 0 0\" ]\n"
    )
    result = subprocess.run(
        ["bash", "-s"],
        input=script,
        env={
            **os.environ,
            "CAPTURE": str(capture),
            "MODEL": "claude-opus-4-6",
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SCIENTIST_STREAM": str(stream),
            "STREAM_REDACTOR": str(REPO / "rollout" / "redact_claude_stream.py"),
            "STUDY_PROMPT_SHA256": expected_sha,
            "STUDY_PROMPT_BYTES": str(len(prompt)),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert capture.read_bytes() == prompt
    assert stream.read_bytes() == prompt
    assert result.stdout.encode() == prompt


def test_claude_stream_redactor_scrubs_credentials_but_keeps_telemetry() -> None:
    redactor = REPO / "rollout" / "redact_claude_stream.py"
    event = {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "content": (
                        "CLAUDE_CODE_MESSAGING_TOKEN=deadbeefdeadbeefdeadbeef\n"
                        "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz123456\n"
                        "Authorization: Bearer bearer-secret-value\n"
                        "safe=value"
                    ),
                }
            ]
        },
        "usage": {"input_tokens": 17, "cache_read_input_tokens": 9},
        "access_token": "persistent-secret-value",
        "signature": "telemetry-signature",
    }
    source = json.dumps(event) + "\n" + "hf_abcdefghijklmnopqrstuvwxyz654321\n"
    result = subprocess.run(
        [sys.executable, str(redactor)],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0 and result.stderr == ""
    assert "deadbeef" not in result.stdout
    assert "abcdefghijklmnopqrstuvwxyz" not in result.stdout
    assert "bearer-secret-value" not in result.stdout
    assert "persistent-secret-value" not in result.stdout
    lines = result.stdout.splitlines()
    clean = json.loads(lines[0])
    assert clean["usage"] == {"input_tokens": 17, "cache_read_input_tokens": 9}
    assert clean["signature"] == "telemetry-signature"
    assert clean["access_token"] == "<redacted>"
    assert clean["message"]["content"][0]["content"].endswith("safe=value")
    assert lines[1] == "<redacted>"


@pytest.mark.parametrize(
    ("metrics", "solve_exit", "expected"),
    [
        (None, 0, 1),
        ("not json", 0, 1),
        ("{}", 0, 1),
        ('{"error": "failed"}', 0, 1),
        ('{"accuracy": true}', 0, 1),
        ('{"accuracy": NaN}', 0, 1),
        ('{"accuracy": 0.625}', 0, 0),
        ('{"accuracy": 0.625}', 124, 124),
    ],
)
def test_study_runner_footer_requires_finite_accuracy_and_preserves_solve_rc(
    tmp_path: Path, metrics: str | None, solve_exit: int, expected: int
) -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_study_runner.py")
    if metrics is not None:
        (tmp_path / "metrics.json").write_text(metrics)
    result = subprocess.run(
        ["bash", "-s"],
        input=(
            "set +e\n"
            f"EVAL_DIR={tmp_path!s}\n"
            f"SOLVE_EXIT={solve_exit}\n"
            + patcher.EOF_REPLACEMENT
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == expected


def test_os_visible_gpu_probe_requires_exactly_one_driver_device(tmp_path: Path) -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_study_runner.py")
    match = re.search(r"python -c '([^']*torch\.cuda\.device_count\(\)[^']*)'", patcher.NEW_SOLVE_LINE)
    assert match is not None
    code = match.group(1)
    fake = tmp_path / "torch.py"
    fake.write_text(
        "import os\n"
        "class cuda:\n"
        "    @staticmethod\n"
        "    def device_count(): return int(os.environ['FAKE_DRIVER_GPU_COUNT'])\n"
    )
    for count, expected in ((0, 86), (1, 0), (2, 86), (8, 86)):
        result = subprocess.run(
            [
                "env",
                "-u",
                "CUDA_VISIBLE_DEVICES",
                "-u",
                "NVIDIA_VISIBLE_DEVICES",
                sys.executable,
                "-c",
                code,
            ],
            env={
                "PATH": os.environ["PATH"],
                "PYTHONPATH": str(tmp_path),
                "FAKE_DRIVER_GPU_COUNT": str(count),
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "0",
            },
            check=False,
        )
        assert result.returncode == expected


def test_study_runner_env_passthrough_is_identifier_only_and_literal(tmp_path: Path) -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_study_runner.py")
    patched = patcher.apply(_pinned_ptb_run_task())
    start = patched.index(patcher.MARK)
    end = patched.index("# Copy scripts needed inside the container", start)
    env_block = patched[start:end]
    agent_dir = tmp_path / "agents" / "demo"
    agent_dir.mkdir(parents=True)
    passthrough = agent_dir / "env_passthrough.txt"
    marker = tmp_path / "must-not-exist"
    literal_value = f"space $(touch {marker}) * ' quote"
    passthrough.write_text("# names only\nSAFE_VALUE\n")
    harness = (
        "set -euo pipefail\n"
        "AGENT=demo\n"
        "EVALUATION_TASK=gsm8k\n"
        + env_block
        + "printf '<%s>\\n' \"${AGENT_ENV_ARGS[@]}\"\n"
    )
    result = subprocess.run(
        ["bash", "-s"],
        input=harness,
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"], "SAFE_VALUE": literal_value},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "<--env>" in result.stdout
    assert f"<SAFE_VALUE={literal_value}>" in result.stdout
    assert not marker.exists()

    passthrough.write_text(f"BAD=$(touch {marker})\n")
    rejected = subprocess.run(
        ["bash", "-s"],
        input=harness,
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"]},
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "invalid variable name" in rejected.stderr
    assert not marker.exists()


def test_study_runner_own_reaper_spares_unattributed_same_uid_process(
    tmp_path: Path,
) -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_study_runner.py")
    patched = patcher.apply(_pinned_ptb_run_task())
    start = patched.index("reap_cell_gpu_processes() {")
    end = patched.index("\nrun_evaluation() {", start)
    reaper = patched[start:end]
    token = "awm-test-cell-token"
    owned_env = os.environ.copy()
    owned_env["POST_TRAIN_BENCH_CELL_TOKEN"] = token
    owned = subprocess.Popen(["sleep", "60"], env=owned_env)
    decoy = subprocess.Popen(["sleep", "60"])
    try:
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_nvidia_smi = fake_bin / "nvidia-smi"
        fake_nvidia_smi.write_text(
            "#!/bin/bash\n"
            f"printf '%s\\n' {owned.pid} {decoy.pid}\n"
        )
        fake_nvidia_smi.chmod(0o755)
        harness = (
            "set -euo pipefail\n"
            + reaper
            + "\nexport POST_TRAIN_BENCH_EVAL_GPU_REAP=own\n"
            + "export POST_TRAIN_BENCH_ISOLATE_GPUS=1\n"
            + "export POST_TRAIN_BENCH_VISIBLE_GPUS=0\n"
            + f"export POST_TRAIN_BENCH_CELL_TOKEN={token}\n"
            + "reap_cell_gpu_processes\n"
        )
        subprocess.run(
            ["bash", "-s"],
            input=harness,
            env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
            text=True,
            capture_output=True,
            check=True,
            timeout=15,
        )
        owned.wait(timeout=5)
        assert decoy.poll() is None
    finally:
        for process in (owned, decoy):
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_study_shell_scripts_parse() -> None:
    scripts = [
        REPO / "rollout" / "setup.sh",
        REPO / "rollout" / "pin_ptb_source.sh",
        REPO / "rollout" / "wm_pack.sbatch",
        REPO / "rollout" / "agents" / "claude_fulltraj_noawm" / "solve.sh",
        REPO / "rollout" / "agents" / "claude_wm" / "solve.sh",
    ]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_source_pin_snapshot_executes_only_snapshot_inputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    for directory in (
        "src/eval/general",
        "src/eval/tasks/gsm8k",
        "agents/hv_recipe",
        "agents/hv_noop",
        "agents/claude_fulltraj_noawm",
        "agents/claude_wm",
    ):
        (source / directory).mkdir(parents=True)
    (source / ".env").write_text("SNAPSHOT_VALUE=original\n")
    (source / "src" / "run_task.sh").write_text(
        "#!/bin/bash\nset -e\npwd\ncat .env\ncat src/eval/general/prompt_fulltraj.txt\n"
        "cat agents/claude_fulltraj_noawm/marker\n"
    )
    (source / "src" / "run_task.sh").chmod(0o755)
    prompts = (
        "prompt_fulltraj.txt",
        "prompt_wm.txt",
        "prompt_wm_fulltraj.txt",
        "prompt_wm_smoke.txt",
        "prompt_wm_fulltraj_smoke.txt",
    )
    for prompt in prompts:
        (source / "src" / "eval" / "general" / prompt).write_text("original prompt\n")
    # PTB's generated test copies are deliberately untracked. The snapshot
    # still has to preserve the setup-attested bytes.
    test_data = source / "src" / "eval" / "tasks" / "gsm8k" / "test_data.json"
    test_data.write_text('[{"question":"q","answer":"a"}]\n')
    (source / ".gitignore").write_text("**/test_data.json\n")
    for agent in ("hv_recipe", "hv_noop", "claude_fulltraj_noawm", "claude_wm"):
        (source / "agents" / agent / "marker").write_text("original agent\n")
        (source / "agents" / agent / "auth.json").write_text("stale tracked auth\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(source), "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
        check=True,
    )
    # Model setup's stage replacement: tracked stale auth is absent from the
    # prepared source and must therefore also be absent from the snapshot.
    for agent in ("hv_recipe", "hv_noop", "claude_fulltraj_noawm", "claude_wm"):
        (source / "agents" / agent / "auth.json").unlink()

    target = tmp_path / "snapshot"
    result = subprocess.run(
        ["bash", str(REPO / "rollout" / "pin_ptb_source.sh"), str(source), str(target)],
        text=True,
        capture_output=True,
        check=True,
    )
    snapshot = Path(result.stdout.strip())
    assert snapshot == target
    assert not list(snapshot.glob("agents/*/auth.json"))
    for prompt in prompts:
        assert (snapshot / "src" / "eval" / "general" / prompt).read_text() == "original prompt\n"
    assert (
        snapshot / "src" / "eval" / "tasks" / "gsm8k" / "test_data.json"
    ).read_bytes() == test_data.read_bytes()

    (source / ".env").write_text("SNAPSHOT_VALUE=mutated\n")
    (source / "src" / "eval" / "general" / "prompt_fulltraj.txt").write_text("mutated prompt\n")
    (source / "agents" / "claude_fulltraj_noawm" / "marker").write_text("mutated agent\n")
    test_data.write_text('[{"question":"mutated","answer":"fixture"}]\n')
    executed = subprocess.run(
        ["bash", "src/run_task.sh"], cwd=snapshot, text=True, capture_output=True, check=True
    )
    assert str(snapshot) in executed.stdout
    assert "SNAPSHOT_VALUE=original" in executed.stdout
    assert "original prompt" in executed.stdout and "original agent" in executed.stdout
    assert "mutated" not in executed.stdout
    assert (
        snapshot / "src" / "eval" / "tasks" / "gsm8k" / "test_data.json"
    ).read_text() == '[{"question":"q","answer":"a"}]\n'


def test_claude_agents_use_vertex_passthrough_without_oauth() -> None:
    agent_root = REPO / "rollout" / "agents"
    required_vertex = {
        "CLAUDE_CODE_USE_VERTEX",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "VERTEX_REGION_CLAUDE_4_6_OPUS",
        "VERTEX_REGION_CLAUDE_4_8_OPUS",
        "VERTEX_REGION_CLAUDE_5_OPUS",
        "AWM_STUDY_CONDITION",
        "AWM_STUDY_REPETITION",
        "AWM_STUDY_MODE",
        "AWM_STUDY_NUM_HOURS",
        "AWM_PRIOR_CORPUS_MANIFEST_SHA256",
        "AWM_PTB_SURFACE_MANIFEST_SHA256",
        "AWM_EXPECTED_SCIENTIST_MODEL_ID",
        "AWM_CLAUDE_CLI_VERSION",
        "AWM_EXPECTED_CLAUDE_CLI_VERSION_OUTPUT",
    }
    for agent in ("claude_fulltraj_noawm", "claude_wm"):
        solve = (agent_root / agent / "solve.sh").read_text()
        names = {
            line.strip()
            for line in (agent_root / agent / "env_passthrough.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        assert all(name.isidentifier() for name in names)
        assert required_vertex <= names
        assert "oauth_token" not in solve
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in names
        assert "CLAUDE_CODE_USE_VERTEX" in solve
        assert "ANTHROPIC_VERTEX_PROJECT_ID" in solve
        assert "metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" in solve
        assert "Vertex needs an attached Google service account" in solve
        assert "persistent ADC files are forbidden" in solve
        for secret in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
            assert secret in solve
        assert json.loads((agent_root / agent / "api_keys.json").read_text()) == {
            "allowed_api_keys": []
        }
    wm_names = (agent_root / "claude_wm" / "env_passthrough.txt").read_text()
    assert "AWM_REPO_COMMIT" in wm_names and "AWM_REPO_URL" not in wm_names
    assert "AWM_CARD_CORPUS_MANIFEST_SHA256" in wm_names
    setup = (REPO / "rollout" / "setup.sh").read_text()
    patch_call = 'python3 "$HERE/patches/apply_study_runner.py"'
    probe = "pinned PTB runner lacks agents/<agent>/env_passthrough.txt support"
    assert patch_call in setup and setup.index(patch_call) < setup.index(probe)
    assert 'python3 "$HERE/patches/apply_extra_binds.py"' in setup
    assert 'python3 "$HERE/patches/apply_scratch_root.py"' in setup
    assert 'env_passthrough.txt" "$DST/agents/$a/env_passthrough.txt"' in setup
    assert "pinned PTB runner lacks agents/<agent>/env_passthrough.txt support" in setup
    assert "pinned PTB prompt loader cannot select the study prompts" in setup
    assert "pinned PTB runner lacks required study capability" in setup
    for capability in (
        "SOLVE_RC",
        "solve_exit_code.txt",
        "POST_TRAIN_BENCH_VISIBLE_GPUS",
        "POST_TRAIN_BENCH_ISOLATE_GPUS",
        "POST_TRAIN_BENCH_EVAL_GPU_REAP",
        "POST_TRAIN_BENCH_TMP_ROOT",
        "awm_cleanup_owned_scratch",
    ):
        assert capability in setup
    assert 'archive --format=tar "$AWM_REPO_COMMIT"' in setup
    for packaged in (
        "rollout/validate_study_corpus.py",
        "rollout/validate_base_model_cache.py",
        "rollout/validate_c1_final_model.py",
        "rollout/attest_claude_runtime.py",
    ):
        assert packaged in setup
    assert setup.count('validate_study_corpus.py"') >= 2
    assert setup.count('validate_base_model_cache.py"') >= 3
    assert setup.count('validate_c1_final_model.py"') >= 4
    assert setup.count('attest_claude_runtime.py"') >= 2
    assert setup.count('validate_wma_session.py"') >= 2
    assert setup.count('redact_claude_stream.py"') >= 3
    assert setup.count('sanitize_result_tree.py"') >= 3
    assert 'install -m 0755 "$HERE/pin_ptb_source.sh"' in setup
    assert 'install -m 0755 "$HERE/attest_ptb_surface.py"' in setup
    assert 'printf \'%s\\n\' "${AWM_REPO_COMMIT,,}"' in setup
    assert "results/ or the historical-card corpus" in setup
    assert 'OAUTH="$SRC/' not in setup


@pytest.mark.parametrize(
    "secret", ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
)
def test_vertex_solve_rejects_injected_direct_claude_credentials(
    secret: str,
) -> None:
    solve = (REPO / "rollout" / "agents" / "claude_fulltraj_noawm" / "solve.sh").read_text()
    start = solve.index("for secret_name in ANTHROPIC_API_KEY")
    end = solve.index('\n\n[ "${CLAUDE_CODE_USE_VERTEX:-}"', start)
    block = solve[start:end]
    marker = "value-must-not-be-logged"
    result = subprocess.run(
        ["bash", "-s"],
        input="set -uo pipefail\n" + block + "\necho reached\n",
        env={"PATH": os.environ["PATH"], secret: marker},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert secret in result.stderr and marker not in result.stderr


def test_claude_agents_propagate_failure_and_require_submission() -> None:
    agent_root = REPO / "rollout" / "agents"
    for agent in ("claude_fulltraj_noawm", "claude_wm"):
        solve = (agent_root / agent / "solve.sh").read_text()
        assert 'pipeline_status=("${PIPESTATUS[@]}")' in solve
        assert 'cat "${STUDY_PROMPT_FILE}" | claude --print' in solve
        assert "verify_study_prompt" in solve
        assert '"$PROMPT"' not in solve
        assert "STUDY_PROMPT_SHA256" in solve
        assert "STUDY_PROMPT_BYTES" in solve
        assert "claude-exit-code.txt" in solve
        assert "attest_claude_runtime.py" in solve
        assert "scientist-model-attestation.json" in solve
        assert "claude-cli-attestation.json" in solve
        assert "install-cli" in solve
        assert "AWM_CLAUDE_CLI_VERSION" in solve
        assert "Claude exited successfully without a non-empty final_model/" in solve
        assert "find /home/ben/task/final_model -mindepth 1 -print -quit" in solve
        assert "validate_study_corpus.py" in solve
        assert "redact_claude_stream.py" in solve
        assert 'python3 "${STREAM_REDACTOR}"' in solve
        assert '--capture "${SCIENTIST_STREAM}"' in solve
        assert '| tee "${SCIENTIST_STREAM}"' not in solve
        assert "redactor_rc" in solve
        assert "tee_rc" not in solve
        assert "sanitize_result_tree.py" in solve
        assert 'python3 "${RESULT_SANITIZER}" /home/ben/task' in solve
        assert "quarantining this cell" in solve
        assert "--require-readonly" in solve
        assert "--record /home/ben/task/study-input.json" in solve
    wm_solve = (agent_root / "claude_wm" / "solve.sh").read_text()
    c1_solve = (agent_root / "claude_fulltraj_noawm" / "solve.sh").read_text()
    assert "validate_c1_final_model.py" in c1_solve
    assert "c1-final-model-attestation.json" in c1_solve
    assert "--expected-base-model google/gemma-3-4b-pt" in c1_solve
    assert "--expected-base-revision" in c1_solve
    assert "--expected-base-checkpoint" in c1_solve
    assert (
        "/home/ben/pinned-base/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
        in c1_solve
    )
    assert "validate_wma_session.py" in wm_solve
    assert "wma-session-attestation.json" in wm_solve
    assert "validate_c1_final_model.py" in wm_solve
    assert "wma-final-model-attestation.json" in wm_solve
    assert "--expected-base-model google/gemma-3-4b-pt" in wm_solve
    assert "--expected-base-revision" in wm_solve
    assert "--expected-base-checkpoint" in wm_solve
    assert (
        "/home/ben/pinned-base/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
        in wm_solve
    )
    validator_call = wm_solve[wm_solve.index('python3 "${WMA_VALIDATOR}"') :]
    assert "--expected-base-model google/gemma-3-4b-pt" in validator_call
    assert '--expected-base-checkpoint "${BASE_MODEL_CHECKPOINT}"' in validator_call
    final_model_validator_call = wm_solve[
        wm_solve.index('python3 "${FINAL_MODEL_VALIDATOR}"') :
    ]
    assert "/home/ben/task/final_model" in final_model_validator_call
    assert "--expected-base-model google/gemma-3-4b-pt" in final_model_validator_call
    assert '--expected-base-revision "${BASE_MODEL_REVISION}"' in final_model_validator_call
    assert '--expected-base-checkpoint "${BASE_MODEL_CHECKPOINT}"' in final_model_validator_call
    assert "--study-input /home/ben/task/study-input.json" in final_model_validator_call
    assert "--record /home/ben/task/wma-final-model-attestation.json" in (
        final_model_validator_call
    )
    assert "WMA_VALIDATOR_ARGS" not in wm_solve


def test_wm_agent_separates_peer_raw_from_peer_cards_and_pins_models() -> None:
    solve = (REPO / "rollout" / "agents" / "claude_wm" / "solve.sh").read_text()
    assert "--memory-readonly" in solve and "--split-side test" in solve
    assert "C2 requires arm=traj" in solve
    assert "C2 must not receive historical card memory" in solve
    assert "--prior-runs /home/ben/prior_runs" in solve
    assert "C3 requires arm=retrieval" in solve
    assert "C3 requires seeded card memory" in solve
    assert "--memory-root /home/ben/wm-memory" in solve
    assert '--wma-model "${AWM_WMA_MODEL}"' in solve
    assert "readonly WMA_REQUESTED_ALIAS=claude-opus-5" in solve
    assert 'claude --print --verbose --model "${WMA_REQUESTED_ALIAS}"' in solve
    assert '--requested-alias "${WMA_REQUESTED_ALIAS}"' in solve
    assert "AWM_REPO_REF" not in solve and "--branch" not in solve
    assert "AWM_REPO_COMMIT" in solve
    assert "^[0-9a-fA-F]{40}$" in solve
    assert "AWM_SRC=/home/ben/agent/awm-src" in solve
    assert 'AWM_SHA="$(tr -d \'[:space:]\' < "${AWM_SRC}/AWM_COMMIT")"' in solve
    assert "git fetch" not in solve and "git clone" not in solve
    assert '[ "${AWM_SHA}" = "${AWM_REPO_COMMIT,,}" ]' in solve
    assert "ListAgents,SendMessage" in solve
    assert "wma-session.jsonl" in solve
    assert "--study-input-key wma_model" in solve
    assert "validate_wma_session.py" in solve


def test_pack_uses_explicit_conditions_and_no_site_slurm_config() -> None:
    pack = (REPO / "rollout" / "wm_pack.sbatch").read_text()
    assert "#SBATCH" not in pack
    assert "/rmeng_data/" not in pack
    assert "SLURM_JOB_ID" not in pack.replace('${SLURM_JOB_ID:-local-$$}', "")
    assert 'RUN_ID="${PTB_RUN_ID:-${SLURM_JOB_ID:-local-$$}}"' in pack
    assert 'GPU_SLOTS_RAW="${PTB_GPU_SLOTS:-${CUDA_VISIBLE_DEVICES:-}}"' in pack
    assert 'POST_TRAIN_BENCH_VISIBLE_GPUS="${GPU_SLOTS[$gpu]}"' in pack
    assert 'POST_TRAIN_BENCH_CUDA_VISIBLE_DEVICES="${GPU_SLOTS[$gpu]}"' in pack
    assert "persistent ADC file credentials are forbidden" in pack
    assert "vertex_auth=attached-service-account" in pack
    assert "PRIOR_RUNS_FOR" not in pack
    assert 'c1:<model>:<prior-scope>:<rep>' in pack
    assert 'c2:<model>:traj:<prior-scope>:<rep>' in pack
    assert 'c3:<model>:retrieval:<memory-sides>:<rep>' in pack
    assert 'config="${scientist}:${arm}:${declared_sides}"' in pack
    assert '${PRIOR_RUNS}:/home/ben/prior_runs:ro' in pack
    assert '${WM_MEMORY}:/home/ben/wm-memory:ro' in pack
    assert 'export AWM_STUDY_CONDITION="${condition}"' in pack
    assert 'export AWM_STUDY_REPETITION="${repetition}"' in pack
    assert 'export AWM_EXPECTED_SCIENTIST_MODEL_ID="${EXPECTED_SCIENTIST_MODELS[$gpu]}"' in pack
    assert "AWM_SCIENTIST_MODEL_ID_4_6" not in pack
    assert "AWM_SCIENTIST_MODEL_ID_4_8" in pack
    assert "AWM_SCIENTIST_MODEL_ID_5_0" in pack
    assert "GPU safety requires exactly one cell per one-GPU invocation" in pack
    assert "GPU safety requires exactly one declared GPU slot" in pack
    assert 'CELL_IDS+=("${RUN_ID}_${STUDY_MODE}_${safe_spec}")' in pack
    assert "refusing to overwrite existing study result" in pack
    assert 'mkdir "${result_dir}"' in pack
    assert "AWM_PRIOR_CORPUS_MANIFEST_SHA256" in pack
    assert "AWM_CARD_CORPUS_MANIFEST_SHA256" in pack
    assert "AWM_PTB_SURFACE_MANIFEST_SHA256" in pack
    assert "attest_study_surface.py" in pack
    assert "POST_TRAIN_BENCH_ISOLATE_GPUS=1" in pack
    assert "POST_TRAIN_BENCH_EVAL_GPU_REAP=own or none" in pack
    assert "validate_base_model_cache.py" in pack
    assert "BASE_MODEL_REVISION=cc012e0a6d0787b4adcc0fa2c4da74402494554d" in pack
    assert "implicit container environment injection is forbidden" in pack
    assert '"${MODEL_CACHE}:/home/ben/pinned-base:ro"' in pack
    assert 'cd "${PINNED_REPO}"' in pack
    assert "bash src/run_task.sh" in pack
    bind_section = pack.split('binds=("${MODEL_CACHE}:/home/ben/pinned-base:ro")', 1)[1]
    c2_binds = bind_section.split("c2)", 1)[1].split(";;", 1)[0]
    c3_binds = bind_section.split("c3)", 1)[1].split(";;", 1)[0]
    assert "prior_runs:ro" in c2_binds and "wm-memory" not in c2_binds
    assert "wm-memory:ro" in c3_binds and "prior_runs" not in c3_binds
    readme = (REPO / "rollout" / "README.md").read_text()
    assert "2 scientist models × 3 information conditions × 2 prior scopes" in readme
    assert "× 2\nexplicit repetitions" in readme
    assert "**24 cells**" in readme
    assert "serves the single\n`consult` verb" in readme
    assert "C3 searches the complete reconstructed-card memory" in readme
    assert "not an operating-system security" in readme


def test_setup_requires_local_paths_and_has_no_machine_defaults() -> None:
    setup = (REPO / "rollout" / "setup.sh").read_text()
    for name in ("PTB_SOURCE_DIR", "HV_PTB_DIR", "PTB_RESULTS_DIR"):
        assert f'${{{name}:?' in setup
    assert "/rmeng_data/" not in setup
    assert "#SBATCH" not in setup
    assert "private PTB .env must set POST_TRAIN_BENCH_ISOLATE_GPUS=1" in setup
    assert "private PTB .env must set POST_TRAIN_BENCH_EVAL_GPU_REAP=own or none" in setup
    pack = (REPO / "rollout" / "wm_pack.sbatch").read_text()
    assert "export POST_TRAIN_BENCH_SKIP_CLI_UPDATE=1" in pack
    assert "export POST_TRAIN_BENCH_JUDGE_AUTH_MODE=skip" in pack
    assert '${POST_TRAIN_BENCH_TMP_ROOT:?' in pack
    assert 'POST_TRAIN_BENCH_TMP_ROOT:-/tmp' not in pack
    assert 'POST_TRAIN_BENCH_TMP_ROOT must be unaliased, owned by this uid' in pack
    assert pack.index('POST_TRAIN_BENCH_TMP_ROOT must be unaliased') < pack.index(
        'pin_src_locally.sh "$REPO_ROOT"'
    )


def test_setup_rejects_source_as_destination_before_mutation(tmp_path: Path) -> None:
    same = tmp_path / "same"
    same.mkdir()
    result = subprocess.run(
        ["bash", str(REPO / "rollout" / "setup.sh")],
        env={
            "PATH": os.environ["PATH"],
            "PTB_SOURCE_DIR": str(same),
            "HV_PTB_DIR": str(same),
            "PTB_RESULTS_DIR": str(tmp_path / "results"),
            "AWM_REPO_COMMIT": "a" * 40,
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "must be a private checkout" in result.stderr
    assert list(same.iterdir()) == []


def test_setup_fresh_private_clone_is_idempotent_and_removes_stale_auth(tmp_path: Path) -> None:
    source_submodule = REPO / "third_party" / "PostTrainBench"
    if not (source_submodule / ".git").exists():
        pytest.skip("PostTrainBench submodule not checked out")
    source = tmp_path / "ptb-source"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", str(source_submodule), str(source)],
        check=True,
    )
    results = tmp_path / "results"
    (source / ".env").write_text(
        "\n".join(
            (
                f'POST_TRAIN_BENCH_RESULTS_DIR="{tmp_path / "source-results"}"',
                f'POST_TRAIN_BENCH_CONTAINERS_DIR="{tmp_path / "containers"}"',
                'POST_TRAIN_BENCH_CONTAINER_NAME="standard"',
                "POST_TRAIN_BENCH_ISOLATE_GPUS=1",
                "POST_TRAIN_BENCH_EVAL_GPU_REAP=none",
                "",
            )
        )
    )
    test_data = source / "src" / "eval" / "tasks" / "gsm8k" / "test_data.json"
    test_data.parent.mkdir(parents=True, exist_ok=True)
    test_data.write_text('[{"question":"q","answer":"a"}]\n')

    awm_source = tmp_path / "awm-source"
    awm_source.mkdir()
    subprocess.run(["git", "init", "-q", str(awm_source)], check=True)
    for directory in ("awm", "input", ".claude"):
        shutil.copytree(REPO / directory, awm_source / directory)
    shutil.copytree(REPO / "rollout", awm_source / "rollout")
    subprocess.run(["git", "-C", str(awm_source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(awm_source),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    awm_commit = subprocess.check_output(
        ["git", "-C", str(awm_source), "rev-parse", "HEAD"], text=True
    ).strip()
    private = tmp_path / "private"
    env = {
        "PATH": os.environ["PATH"],
        "PTB_SOURCE_DIR": str(source),
        "HV_PTB_DIR": str(private),
        "PTB_RESULTS_DIR": str(results),
        "AWM_SOURCE_DIR": str(awm_source),
        "AWM_REPO_COMMIT": awm_commit,
    }
    first = subprocess.run(
        ["bash", str(REPO / "rollout" / "setup.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    for name in (
        "validate_study_corpus.py",
        "validate_base_model_cache.py",
        "validate_c1_final_model.py",
        "attest_claude_runtime.py",
        "validate_wma_session.py",
        "redact_claude_stream.py",
        "sanitize_result_tree.py",
    ):
        assert (private / "agents" / "claude_wm" / "payload" / name).is_file()
    assert (private / "agents" / "claude_fulltraj_noawm" / "payload" / "attest_claude_runtime.py").is_file()
    assert (
        private / "agents" / "claude_fulltraj_noawm" / "payload" / "validate_base_model_cache.py"
    ).is_file()
    assert (
        private / "agents" / "claude_fulltraj_noawm" / "payload" / "validate_c1_final_model.py"
    ).is_file()
    assert (private / "agents" / "claude_fulltraj_noawm" / "payload" / "redact_claude_stream.py").is_file()
    assert (
        private / "agents" / "claude_fulltraj_noawm" / "payload" / "sanitize_result_tree.py"
    ).is_file()
    assert (private / "src" / "commit_utils" / "pin_src_locally.sh").is_file()
    assert (
        private / "src" / "eval" / "tasks" / "gsm8k" / "test_data.json"
    ).read_bytes() == test_data.read_bytes()
    surface_manifest = private / ".git" / "awm-study-surface.json"
    assert surface_manifest.is_file()

    stale_names = ("auth.json", "cursor_auth.json", "grok_auth.json", "oauth_token")
    for agent in ("claude_fulltraj_noawm", "claude_wm"):
        for name in stale_names:
            (private / "agents" / agent / name).write_text("stale credential\n")
    changed_surface = subprocess.run(
        [
            sys.executable,
            str(private / "src" / "commit_utils" / "attest_study_surface.py"),
            "verify",
            "--root",
            str(private),
            "--manifest",
            str(surface_manifest),
            "--awm-commit",
            awm_commit,
            "--ptb-commit",
            subprocess.check_output(
                ["git", "-C", str(private), "rev-parse", "HEAD"], text=True
            ).strip(),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert changed_surface.returncode == 2
    assert "differs from its setup manifest" in changed_surface.stderr
    second = subprocess.run(
        ["bash", str(REPO / "rollout" / "setup.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    for agent in ("claude_fulltraj_noawm", "claude_wm"):
        assert not any((private / "agents" / agent / name).exists() for name in stale_names)

    snapshot = tmp_path / "snapshot"
    pinned = subprocess.run(
        [
            "bash",
            str(private / "src" / "commit_utils" / "pin_src_locally.sh"),
            str(private),
            str(snapshot),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert pinned.returncode == 0, pinned.stderr
    assert Path(pinned.stdout.strip()) == snapshot
    assert not any(snapshot.glob("agents/claude_*/auth.json"))
    assert (
        snapshot / "src" / "eval" / "tasks" / "gsm8k" / "test_data.json"
    ).read_bytes() == test_data.read_bytes()

    # A different claimed commit may exist locally, but setup must reject it
    # when any bootstrap byte differs from the files it is actually executing.
    with (awm_source / "rollout" / "build_prompts.py").open("a") as handle:
        handle.write("\n# deliberately different committed bootstrap\n")
    subprocess.run(["git", "-C", str(awm_source), "add", "rollout/build_prompts.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(awm_source),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "mismatched fixture",
        ],
        check=True,
    )
    mismatched_commit = subprocess.check_output(
        ["git", "-C", str(awm_source), "rev-parse", "HEAD"], text=True
    ).strip()
    mismatch = subprocess.run(
        ["bash", str(REPO / "rollout" / "setup.sh")],
        env={**env, "AWM_REPO_COMMIT": mismatched_commit},
        text=True,
        capture_output=True,
        check=False,
    )
    assert mismatch.returncode == 2
    assert "does not byte-match AWM_REPO_COMMIT: rollout/build_prompts.py" in mismatch.stderr


def _pack_preflight(tmp_path: Path, *specs: str, extra: dict[str, str] | None = None):
    harness = tmp_path / "harness"
    pack = harness / "rollout" / "wm_pack.sbatch"
    if not pack.exists():
        pack.parent.mkdir(parents=True)
        shutil.copy2(REPO / "rollout" / "wm_pack.sbatch", pack)
        subprocess.run(["git", "init", "-q", str(harness)], check=True)
        subprocess.run(["git", "-C", str(harness), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(harness),
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "pack fixture",
            ],
            check=True,
        )
    harness_commit = subprocess.check_output(
        ["git", "-C", str(harness), "rev-parse", "HEAD"], text=True
    ).strip()
    env = {
        "PATH": os.environ["PATH"],
        "HV_PTB_DIR": str(tmp_path / "ptb"),
        "CLAUDE_CODE_USE_VERTEX": "1",
        "ANTHROPIC_VERTEX_PROJECT_ID": "project",
        "AWM_SCIENTIST_MODEL_ID_4_8": "provider-opus-4-8",
        "AWM_SCIENTIST_MODEL_ID_5_0": "provider-opus-5-0",
        "AWM_CLAUDE_CLI_VERSION": "2.1.251",
        "AWM_EXPECTED_CLAUDE_CLI_VERSION_OUTPUT": "2.1.251 (Claude Code)",
        "AWM_REPO_COMMIT": harness_commit,
    }
    env.update(extra or {})
    return subprocess.run(
        ["bash", str(pack), *specs],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_pack_fails_closed_for_missing_condition_mounts(tmp_path: Path) -> None:
    c1 = _pack_preflight(tmp_path, "c1:claude-opus-4-8:train:1")
    assert c1.returncode == 2 and "C1/C2 requires PRIOR_RUNS" in c1.stderr

    c3 = _pack_preflight(tmp_path, "c3:claude-opus-4-8:retrieval:train:1")
    assert c3.returncode == 2 and "requires seeded WM_MEMORY" in c3.stderr


def test_pack_rejects_removed_opus_4_6_scientist(tmp_path: Path) -> None:
    result = _pack_preflight(tmp_path, "c1:claude-opus-4-6:train:1")
    assert result.returncode == 2
    assert "unsupported scientist model" in result.stderr


def test_pack_requires_two_explicit_unique_repetitions(tmp_path: Path) -> None:
    missing = _pack_preflight(tmp_path, "c1:claude-opus-4-8:train")
    assert missing.returncode == 2 and "<1|2>" in missing.stderr

    out_of_range = _pack_preflight(tmp_path, "c3:claude-opus-5:retrieval:train:3")
    assert out_of_range.returncode == 2 and "<1|2>" in out_of_range.stderr

    duplicate = _pack_preflight(
        tmp_path,
        "c1:claude-opus-4-8:train:1",
        "c1:claude-opus-4-8:train:1",
    )
    assert duplicate.returncode == 2 and "exactly one cell" in duplicate.stderr

    two_reps = _pack_preflight(
        tmp_path,
        "c1:claude-opus-4-8:train:1",
        "c1:claude-opus-4-8:train:2",
    )
    assert two_reps.returncode == 2 and "exactly one cell" in two_reps.stderr

    models = ("claude-opus-4-8", "claude-opus-5")
    conditions = ("c1", "c2", "c3")
    scopes = ("train", "train,test")
    repetitions = (1, 2)
    matrix = {
        (condition, model, scope, repetition)
        for model in models
        for condition in conditions
        for scope in scopes
        for repetition in repetitions
    }
    assert len(matrix) == 24
    pack = (REPO / "rollout" / "wm_pack.sbatch").read_text()
    assert 'CELL_IDS+=("${RUN_ID}_${STUDY_MODE}_${safe_spec}")' in pack


def test_study_matrix_emits_and_validates_exact_24_cells() -> None:
    script = REPO / "rollout" / "study_matrix.py"
    emitted = subprocess.run(
        [sys.executable, str(script)], text=True, capture_output=True, check=True
    )
    records = json.loads(emitted.stdout)
    specs = [record["spec"] for record in records]

    assert len(records) == len(set(specs)) == 24
    assert {record["condition"] for record in records} == {"c1", "c2", "c3"}
    assert {record["scientist_model"] for record in records} == {
        "claude-opus-4-8",
        "claude-opus-5",
    }
    assert {record["scope"] for record in records} == {"train", "train,test"}
    assert {record["repetition"] for record in records} == {1, 2}
    assert {record["benchmark"] for record in records} == {"gsm8k"}
    assert {record["base_model"] for record in records} == {"google/gemma-3-4b-pt"}
    assert {record["num_hours"] for record in records} == {10}
    assert {record["study_mode"] for record in records} == {"production"}
    assert {record["prior_rollout_count"] for record in records} == {143, 193}
    assert {record["includes_gemma_trajectories"] for record in records} == {
        False,
        True,
    }
    assert {
        record["condition"]: record["wma_arm"] for record in records
    } == {"c1": None, "c2": "traj", "c3": "retrieval"}
    assert all(":traj:" in record["spec"] for record in records if record["condition"] == "c2")
    assert all(
        ":retrieval:" in record["spec"]
        for record in records
        if record["condition"] == "c3"
    )

    valid = subprocess.run(
        [sys.executable, str(script), "--validate", *specs],
        text=True,
        capture_output=True,
        check=False,
    )
    assert valid.returncode == 0
    assert json.loads(valid.stdout) == {"cell_count": 24, "valid": True}

    invalid = subprocess.run(
        [sys.executable, str(script), "--validate", *specs[:-1], specs[0]],
        text=True,
        capture_output=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert '"duplicates"' in invalid.stderr and '"missing"' in invalid.stderr


def test_pack_direct_mode_requires_explicit_gpu_slot_not_slurm(tmp_path: Path) -> None:
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "INDEX.md").write_text("# prior\n")
    (prior / "index.jsonl").write_text(json.dumps({"side": "train"}) + "\n")
    result = _pack_preflight(
        tmp_path,
        "c1:claude-opus-4-8:train:1",
        extra={
            "PRIOR_RUNS": str(prior),
            "AWM_PRIOR_CORPUS_MANIFEST_SHA256": "0" * 64,
        },
    )
    assert result.returncode == 2
    assert "set PTB_GPU_SLOTS or launch with CUDA_VISIBLE_DEVICES set" in result.stderr


def test_pack_requires_one_declared_gpu_slot(tmp_path: Path) -> None:
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "INDEX.md").write_text("# prior\n")
    (prior / "index.jsonl").write_text(json.dumps({"side": "train"}) + "\n")
    result = _pack_preflight(
        tmp_path,
        "c1:claude-opus-4-8:train:1",
        extra={
            "PRIOR_RUNS": str(prior),
            "AWM_PRIOR_CORPUS_MANIFEST_SHA256": "0" * 64,
            "PTB_GPU_SLOTS": "0,1",
        },
    )
    assert result.returncode == 2
    assert "exactly one declared GPU slot" in result.stderr


def test_pack_fixes_every_wma_to_verified_opus_5_vertex_identity(
    tmp_path: Path,
) -> None:
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "INDEX.md").write_text("# prior\n")
    (prior / "index.jsonl").write_text(json.dumps({"side": "train"}) + "\n")
    common = {
        "PRIOR_RUNS": str(prior),
        "AWM_PRIOR_CORPUS_MANIFEST_SHA256": "0" * 64,
    }
    wrong = _pack_preflight(
        tmp_path,
        "c2:claude-opus-4-8:traj:train:1",
        extra={**common, "AWM_WMA_MODEL": "provider-opus-4-8"},
    )
    assert wrong.returncode == 2
    assert "every WMA must use the verified Claude Code Opus 5 Vertex identity" in (
        wrong.stderr
    )

    opus_5 = _pack_preflight(
        tmp_path,
        "c2:claude-opus-4-8:traj:train:1",
        extra={**common, "AWM_WMA_MODEL": "provider-opus-5-0"},
    )
    assert opus_5.returncode == 2
    assert "set PTB_GPU_SLOTS or launch with CUDA_VISIBLE_DEVICES set" in opus_5.stderr


def test_pack_smoke_mode_is_explicit_one_hour_and_nonproduction(tmp_path: Path) -> None:
    production_short = _pack_preflight(
        tmp_path,
        "c1:claude-opus-4-8:train:1",
        extra={"PTB_NUM_HOURS": "1"},
    )
    assert production_short.returncode == 2
    assert "production study cells require PTB_NUM_HOURS=10" in production_short.stderr

    smoke_wrong = _pack_preflight(
        tmp_path,
        "c1:claude-opus-4-8:train:1",
        extra={"AWM_STUDY_SMOKE": "1", "PTB_NUM_HOURS": "0.25"},
    )
    assert smoke_wrong.returncode == 2
    assert "AWM_STUDY_SMOKE=1 requires explicit PTB_NUM_HOURS=1" in smoke_wrong.stderr

    smoke = _pack_preflight(
        tmp_path,
        "c1:claude-opus-4-8:train:1",
        extra={"AWM_STUDY_SMOKE": "1", "PTB_NUM_HOURS": "1"},
    )
    assert smoke.returncode == 2
    assert "C1/C2 requires PRIOR_RUNS" in smoke.stderr
    pack = (REPO / "rollout" / "wm_pack.sbatch").read_text()
    assert 'CELL_IDS+=("${RUN_ID}_${STUDY_MODE}_${safe_spec}")' in pack
    assert 'export AWM_STUDY_MODE="${STUDY_MODE}"' in pack
    assert 'export AWM_STUDY_NUM_HOURS="${NUM_HOURS}"' in pack


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"PTB_TASK": "other"}, "requires PTB_TASK=gsm8k"),
        ({"PTB_MODEL": "other/model"}, "requires PTB_MODEL=google/gemma-3-4b-pt"),
    ],
)
def test_pack_rejects_relabeled_task_or_base_model(
    tmp_path: Path, extra: dict[str, str], message: str
) -> None:
    result = _pack_preflight(tmp_path, "c1:claude-opus-4-8:train:1", extra=extra)
    assert result.returncode == 2 and message in result.stderr


def test_pack_rejects_mutable_ref_and_side_mismatch(tmp_path: Path) -> None:
    memory = tmp_path / "memory" / "structured"
    memory.mkdir(parents=True)
    (memory / "cards.jsonl").write_text(
        json.dumps({"provenance": {"split_side": "train"}}) + "\n"
    )
    base = {
        "WM_MEMORY": str(memory.parent),
        "AWM_REPO_COMMIT": "wm-runtime",
        "AWM_CARD_CORPUS_MANIFEST_SHA256": "0" * 64,
        "AWM_WMA_MODEL": "provider-wma-exact",
    }
    mutable = _pack_preflight(tmp_path, "c3:claude-opus-5:retrieval:train:1", extra=base)
    assert mutable.returncode == 2 and "full 40-hex" in mutable.stderr

    both = _pack_preflight(
        tmp_path,
        "c3:claude-opus-5:retrieval:train,test:1",
        extra={
            "WM_MEMORY": str(memory.parent),
            "AWM_CARD_CORPUS_MANIFEST_SHA256": "0" * 64,
        },
    )
    assert both.returncode == 2 and "requires test-side cards" in both.stderr

    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "INDEX.md").write_text("# prior\n")
    (prior / "index.jsonl").write_text(
        json.dumps({"side": "train"}) + "\n" + json.dumps({"side": "test"}) + "\n"
    )
    raw_scope = _pack_preflight(
        tmp_path,
        "c2:claude-opus-4-8:traj:train:1",
        extra={
            "PRIOR_RUNS": str(prior),
            "WM_MEMORY": str(memory.parent),
            "AWM_WMA_MODEL": "claude-opus-5",
            "AWM_PRIOR_CORPUS_MANIFEST_SHA256": "0" * 64,
        },
    )
    assert raw_scope.returncode == 2 and "scope does not match" in raw_scope.stderr
