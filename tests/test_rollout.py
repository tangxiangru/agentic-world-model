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
    audit = session / "wm" / "cards" / "exp-01" / "wma-calls" / "call-1" / "audit.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({"status": "success"}) + "\n")
    events = [
        {"seq": 1, "event": "wma_call", "card_id": "exp-01", "path": str(audit)},
        {"seq": 2, "event": "sealed", "card_id": "exp-01", "checkpoint": "ckpt"},
        {"seq": 3, "event": "adopted", "card_id": "exp-01"},
        {
            "seq": 4,
            "event": "card_closed",
            "card_id": "exp-01",
            "how": "finalize",
            "decision": "adopt",
        },
    ]
    ledger = session / "wm" / "events.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in events))
    submission = session / "final_model"
    submission.mkdir()
    (submission / "config.json").write_text("{}\n")
    return session, events


def test_wma_session_postcondition_requires_successful_call_and_adoption(tmp_path: Path) -> None:
    validator = _load(REPO / "rollout" / "validate_wma_session.py")
    session, _events = _write_valid_wma_session(tmp_path)
    evidence = validator.validate(session)
    assert evidence["adopted_card_ids"] == ["exp-01"]
    assert evidence["successful_wma_call_count"] == 1


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
        events.append(append_event)
    (session / "wm" / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    with pytest.raises(validator.ValidationError, match=message):
        validator.validate(session)


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
    instruction = (REPO / "input" / "instruction.md").read_text()
    wm = bp.wm_prompt(instruction, fulltraj=False)
    for leftover in ("{dir}", "{submission}", "{time_limit}", "{gpu}"):
        assert leftover not in wm
    assert "/home/ben/task/final_model" in wm and "{num_hours} hours" in wm
    assert "{setup_other}{decontamination_tool}" in wm and "{model}" in wm and "{benchmark}" in wm
    assert "## Prior runs" not in wm
    wm_ft = bp.wm_prompt(instruction, fulltraj=True)
    assert wm_ft.index("## Prior runs") < wm_ft.index("## The world-model agent")
    prior = bp.PRIOR_RUNS_SECTION
    for guaranteed in ("`solve_out.txt`", "`metrics.json`", "`time_taken.txt`"):
        assert guaranteed in prior
    assert "Every run directory has exactly" in prior
    assert "Optional upstream artifacts and `task/` workspace snapshots" in prior
    assert "solve_parsed.txt" not in prior
    for name in ("prompt_fulltraj.txt", "prompt_wm_fulltraj.txt"):
        assert prior in (REPO / "rollout" / "prompts" / name).read_text()
    assert prior not in (REPO / "rollout" / "prompts" / "prompt_wm.txt").read_text()
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


def _pinned_ptb_run_task() -> str:
    ptb = REPO / "third_party" / "PostTrainBench"
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
    original = _pinned_ptb_run_task()
    once = patcher.apply(original)
    assert once != original
    assert patcher.apply(once) == once

    combined = extra.apply(once)
    assert extra.apply(combined) == combined
    candidate = tmp_path / "run_task.sh"
    candidate.write_text(combined)
    subprocess.run(["bash", "-n", str(candidate)], check=True)

    assert 'agents/${AGENT}/payload/." "${JOB_DIR}/agent/' in combined
    assert '"${AGENT_ENV_ARGS[@]}" \\' in combined
    assert 'bash -o pipefail -c "{ echo OS-visible GPU isolation probe' in combined
    assert "SOLVE_RC=\\$?" in combined
    assert 'solve_exit_code.txt' in combined
    assert 'exit "$SOLVE_EXIT"' in combined
    assert "skipping optional judges and evaluating the preserved final_model" in combined
    assert 'POST_TRAIN_BENCH_VISIBLE_GPUS' in combined
    assert 'POST_TRAIN_BENCH_ISOLATE_GPUS' in combined
    assert 'POST_TRAIN_BENCH_EVAL_GPU_REAP' in combined
    assert 'POST_TRAIN_BENCH_CELL_TOKEN' in combined
    assert "OS-visible GPU isolation probe" in combined
    assert "env -u CUDA_VISIBLE_DEVICES -u NVIDIA_VISIBLE_DEVICES" in combined
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


def test_study_runner_patch_upgrades_known_earlier_revision() -> None:
    patcher = _load(REPO / "rollout" / "patches" / "apply_study_runner.py")
    current = patcher.apply(_pinned_ptb_run_task())
    older = current.replace(patcher.NEW_SOLVE_LINE, patcher.NEW_SOLVE_LINE_V2)
    older = older.replace(patcher.EOF_REPLACEMENT, patcher.EOF_REPLACEMENT_V2)
    assert patcher.apply(older) == current


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
    for prompt in ("prompt_fulltraj.txt", "prompt_wm.txt", "prompt_wm_fulltraj.txt"):
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
        "GOOGLE_APPLICATION_CREDENTIALS",
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
        assert "Vertex needs ADC or an attached Google service account" in solve
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
    ):
        assert capability in setup
    assert 'archive --format=tar "$AWM_REPO_COMMIT"' in setup
    assert "rollout/validate_study_corpus.py rollout/attest_claude_runtime.py" in setup
    assert setup.count('validate_study_corpus.py"') >= 2
    assert setup.count('attest_claude_runtime.py"') >= 2
    assert setup.count('validate_wma_session.py"') >= 2
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
        assert "claude-exit-code.txt" in solve
        assert "attest_claude_runtime.py" in solve
        assert "scientist-model-attestation.json" in solve
        assert "claude-cli-attestation.json" in solve
        assert "install-cli" in solve
        assert "AWM_CLAUDE_CLI_VERSION" in solve
        assert "Claude exited successfully without a non-empty final_model/" in solve
        assert "find /home/ben/task/final_model -mindepth 1 -print -quit" in solve
        assert "validate_study_corpus.py" in solve
        assert "--require-readonly" in solve
        assert "--record /home/ben/task/study-input.json" in solve
    wm_solve = (agent_root / "claude_wm" / "solve.sh").read_text()
    assert "validate_wma_session.py" in wm_solve
    assert "wma-session-attestation.json" in wm_solve


def test_wm_agent_separates_llm_raw_from_llm_cards_and_pins_models() -> None:
    solve = (REPO / "rollout" / "agents" / "claude_wm" / "solve.sh").read_text()
    assert '[ "${RO:-}" = "ro" ]' in solve
    assert "--memory-readonly --split-side test" in solve
    assert "C2 requires arm=llm" in solve
    assert "C2 must not receive historical card memory" in solve
    assert "MEM=/home/ben/wm-empty-memory" in solve
    assert "--wma-corpus-kind raw --wma-corpus-root /home/ben/prior_runs" in solve
    assert "C3 requires arm=llm" in solve
    assert "C3 requires seeded card memory" in solve
    assert "--wma-corpus-kind cards" in solve
    assert '--wma-model "${AWM_WMA_MODEL}"' in solve
    assert "private empty memory" not in solve
    assert "AWM_REPO_REF" not in solve and "--branch" not in solve
    assert "AWM_REPO_COMMIT" in solve
    assert "^[0-9a-fA-F]{40}$" in solve
    assert "AWM_SRC=/home/ben/agent/awm-src" in solve
    assert 'AWM_SHA="$(tr -d \'[:space:]\' < "${AWM_SRC}/AWM_COMMIT")"' in solve
    assert "git fetch" not in solve and "git clone" not in solve
    assert '[ "${AWM_SHA}" = "${AWM_REPO_COMMIT,,}" ]' in solve
    assert '"agent_(degraded|failed)"' in solve
    assert "censoring this labelled cell" in solve
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
    assert '${VERTEX_ADC_FILE}:/home/ben/.config/gcloud/application_default_credentials.json:ro' in pack
    assert "PRIOR_RUNS_FOR" not in pack
    assert 'c1:<model>:<prior-scope>:<rep>' in pack
    assert 'c2:<model>:llm:<prior-scope>:<rep>' in pack
    assert 'c3:<model>:llm:<memory-sides>:<rep>' in pack
    assert 'config="${scientist}:${arm}:${declared_sides}:ro"' in pack
    assert '${PRIOR_RUNS}:/home/ben/prior_runs:ro' in pack
    assert '${WM_MEMORY}:/home/ben/wm-memory:ro' in pack
    assert 'export AWM_STUDY_CONDITION="${condition}"' in pack
    assert 'export AWM_STUDY_REPETITION="${repetition}"' in pack
    assert 'export AWM_EXPECTED_SCIENTIST_MODEL_ID="${EXPECTED_SCIENTIST_MODELS[$gpu]}"' in pack
    assert "AWM_SCIENTIST_MODEL_ID_4_6" in pack
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
    assert 'cd "${PINNED_REPO}"' in pack
    assert "bash src/run_task.sh" in pack
    bind_section = pack.split("binds=()", 1)[1]
    c2_binds = bind_section.split("c2)", 1)[1].split(";;", 1)[0]
    c3_binds = bind_section.split("c3)", 1)[1].split(";;", 1)[0]
    assert "prior_runs:ro" in c2_binds and "wm-memory" not in c2_binds
    assert "wm-memory:ro" in c3_binds and "prior_runs" not in c3_binds
    readme = (REPO / "rollout" / "README.md").read_text()
    assert "3 scientist models × 3 information conditions × 2 prior scopes" in readme
    assert "× 2\nexplicit repetitions" in readme
    assert "**36 cells**" in readme
    assert "no host-selected top-k" in readme
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
        "attest_claude_runtime.py",
        "validate_wma_session.py",
    ):
        assert (private / "agents" / "claude_wm" / "payload" / name).is_file()
    assert (private / "agents" / "claude_fulltraj_noawm" / "payload" / "attest_claude_runtime.py").is_file()
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
        "AWM_SCIENTIST_MODEL_ID_4_6": "provider-opus-4-6",
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
    c1 = _pack_preflight(tmp_path, "c1:claude-opus-4-6:train:1")
    assert c1.returncode == 2 and "C1/C2 requires PRIOR_RUNS" in c1.stderr

    c3 = _pack_preflight(tmp_path, "c3:claude-opus-4-8:llm:train:1")
    assert c3.returncode == 2 and "requires seeded WM_MEMORY" in c3.stderr


def test_pack_requires_two_explicit_unique_repetitions(tmp_path: Path) -> None:
    missing = _pack_preflight(tmp_path, "c1:claude-opus-4-6:train")
    assert missing.returncode == 2 and "<1|2>" in missing.stderr

    out_of_range = _pack_preflight(tmp_path, "c3:claude-opus-5:llm:train:3")
    assert out_of_range.returncode == 2 and "<1|2>" in out_of_range.stderr

    duplicate = _pack_preflight(
        tmp_path,
        "c1:claude-opus-4-6:train:1",
        "c1:claude-opus-4-6:train:1",
    )
    assert duplicate.returncode == 2 and "exactly one cell" in duplicate.stderr

    two_reps = _pack_preflight(
        tmp_path,
        "c1:claude-opus-4-6:train:1",
        "c1:claude-opus-4-6:train:2",
    )
    assert two_reps.returncode == 2 and "exactly one cell" in two_reps.stderr

    models = ("claude-opus-4-6", "claude-opus-4-8", "claude-opus-5")
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
    assert len(matrix) == 36
    pack = (REPO / "rollout" / "wm_pack.sbatch").read_text()
    assert 'CELL_IDS+=("${RUN_ID}_${STUDY_MODE}_${safe_spec}")' in pack


def test_study_matrix_emits_and_validates_exact_36_cells() -> None:
    script = REPO / "rollout" / "study_matrix.py"
    emitted = subprocess.run(
        [sys.executable, str(script)], text=True, capture_output=True, check=True
    )
    records = json.loads(emitted.stdout)
    specs = [record["spec"] for record in records]

    assert len(records) == len(set(specs)) == 36
    assert {record["condition"] for record in records} == {"c1", "c2", "c3"}
    assert {record["scientist_model"] for record in records} == {
        "claude-opus-4-6",
        "claude-opus-4-8",
        "claude-opus-5",
    }
    assert {record["scope"] for record in records} == {"train", "train,test"}
    assert {record["repetition"] for record in records} == {1, 2}
    assert {record["benchmark"] for record in records} == {"gsm8k"}
    assert {record["base_model"] for record in records} == {"google/gemma-3-4b-pt"}
    assert {record["num_hours"] for record in records} == {10}
    assert {record["study_mode"] for record in records} == {"production"}

    valid = subprocess.run(
        [sys.executable, str(script), "--validate", *specs],
        text=True,
        capture_output=True,
        check=False,
    )
    assert valid.returncode == 0
    assert json.loads(valid.stdout) == {"cell_count": 36, "valid": True}

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
        "c1:claude-opus-4-6:train:1",
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
        "c1:claude-opus-4-6:train:1",
        extra={
            "PRIOR_RUNS": str(prior),
            "AWM_PRIOR_CORPUS_MANIFEST_SHA256": "0" * 64,
            "PTB_GPU_SLOTS": "0,1",
        },
    )
    assert result.returncode == 2
    assert "exactly one declared GPU slot" in result.stderr


def test_pack_smoke_mode_is_explicit_one_hour_and_nonproduction(tmp_path: Path) -> None:
    production_short = _pack_preflight(
        tmp_path,
        "c1:claude-opus-4-6:train:1",
        extra={"PTB_NUM_HOURS": "1"},
    )
    assert production_short.returncode == 2
    assert "production study cells require PTB_NUM_HOURS=10" in production_short.stderr

    smoke_wrong = _pack_preflight(
        tmp_path,
        "c1:claude-opus-4-6:train:1",
        extra={"AWM_STUDY_SMOKE": "1", "PTB_NUM_HOURS": "0.25"},
    )
    assert smoke_wrong.returncode == 2
    assert "AWM_STUDY_SMOKE=1 requires explicit PTB_NUM_HOURS=1" in smoke_wrong.stderr

    smoke = _pack_preflight(
        tmp_path,
        "c1:claude-opus-4-6:train:1",
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
    result = _pack_preflight(tmp_path, "c1:claude-opus-4-6:train:1", extra=extra)
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
    mutable = _pack_preflight(tmp_path, "c3:claude-opus-5:llm:train:1", extra=base)
    assert mutable.returncode == 2 and "full 40-hex" in mutable.stderr

    both = _pack_preflight(
        tmp_path,
        "c3:claude-opus-5:llm:train,test:1",
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
        "c2:claude-opus-4-6:llm:train:1",
        extra={
            "PRIOR_RUNS": str(prior),
            "WM_MEMORY": str(memory.parent),
            "AWM_WMA_MODEL": "claude-opus-5",
            "AWM_PRIOR_CORPUS_MANIFEST_SHA256": "0" * 64,
        },
    )
    assert raw_scope.returncode == 2 and "scope does not match" in raw_scope.stderr
