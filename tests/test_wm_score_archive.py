import json
import shutil
from pathlib import Path

import pytest
from test_wm_run import launches, packed, run_bundle, source  # noqa: F401

from tools.outcome_prediction import wm_run as runner
from tools.outcome_prediction import wm_score_archive as archive


@pytest.fixture
def completed(source, tmp_path, launches, monkeypatch):  # noqa: F811
    bundle = packed(source, tmp_path)
    run_bundle(bundle, tmp_path)
    inventory = tmp_path / "private_inventory" / "inventory.jsonl"
    inventory.parent.mkdir()
    lines = [
        {
            "example_id": f"r0-91/exp-{number:02d}",
            "cell_id": "r0-91",
            "label": {"accuracy": value, "official_metric": {"accuracy": value}},
            "model_input": {"forbidden_to_decode": True},
        }
        for number, value in ((1, 0), (2, 0.5))
    ]
    inventory.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline scoring must never launch/resume an agent")

    monkeypatch.setattr(runner, "execute", forbidden)
    monkeypatch.setattr(runner, "_launch", forbidden)
    monkeypatch.setattr(runner, "auth_metadata", forbidden)
    return {
        "bundle_path": bundle["bundle_path"],
        "bundle_sha256": bundle["bundle_sha256"],
        "execution_dir": tmp_path / "captures",
        "inventory_path": inventory,
        "inventory_sha256": runner.sha(inventory.read_bytes()),
        "output_dir": tmp_path / "scores",
    }


def attempt(options, arm="rpm"):
    return Path(options["execution_dir"]) / runner.sha(b"case-one") / arm


def rewrite(path, value):
    path.chmod(0o600)
    path.write_bytes(runner.encoded(value))


def load(path):
    return json.loads(path.read_text())


def rehash(options, arm="rpm"):
    directory = attempt(options, arm)
    result = load(directory / "result.json")
    result["capture_sha256"] = {
        name: runner._file_sha(directory / name) for name in runner.CAPTURE_FILES
    }
    rewrite(directory / "result.json", result)


def deny_labels(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Label access preceded complete frozen-capture verification")

    monkeypatch.setattr(archive, "_archive_labels", forbidden)


def test_complete_success_scores_zero_and_keeps_private_provenance(completed):
    result = archive.score_archive(**completed)
    assert result["completed_attempts"] == 2
    assert result["paired_valid_cases"] == 1
    assert result["labels_or_scores_in_summary"] is False
    output = completed["output_dir"]
    labels = load(output / "labels.json")
    assert labels == {"case-one": {"r0-91/exp-01": 0, "r0-91/exp-02": 0.5}}
    report = load(output / "report.json")
    assert report["bootstrap_replicates"] == 5000
    assert report["bootstrap_seed"] == 20260905
    assert report["cases"][0]["arms"]["rpm"]["selected_model_accuracy"] == 0
    proof = load(output / "provenance.json")
    assert proof["all_frozen_slots_completed_before_inventory_access"]
    assert proof["attempts_re_normalized_from_preserved_captures"] == 2
    assert not proof["inference_launched"] and not proof["retries_performed"]
    assert all(runner.sha((output / n).read_bytes()) == h for n, h in proof["files_sha256"].items())
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in output.iterdir())


@pytest.mark.parametrize("missing", ["result.json", "capture.json", "transcript.jsonl"])
def test_incomplete_attempt_never_opens_even_invalid_inventory(completed, monkeypatch, missing):
    (attempt(completed) / missing).unlink()
    completed["inventory_path"].write_text("definitely invalid labels")
    deny_labels(monkeypatch)
    with pytest.raises(ValueError, match="incomplete"):
        archive.score_archive(**completed)
    assert not completed["output_dir"].exists()


def test_missing_arm_never_reads_labels(completed, monkeypatch):
    (attempt(completed)).rename(attempt(completed).with_name("extra-attempt"))
    deny_labels(monkeypatch)
    with pytest.raises(ValueError, match="extra"):
        archive.score_archive(**completed)


@pytest.mark.parametrize("where", ["root", "case", "attempt"])
def test_extra_attempt_or_unrecognized_artifact_rejected(completed, monkeypatch, where):
    target = {
        "root": completed["execution_dir"],
        "case": attempt(completed).parent,
        "attempt": attempt(completed),
    }[where]
    (target / "unrecorded_retry").mkdir()
    deny_labels(monkeypatch)
    with pytest.raises(ValueError, match="extra"):
        archive.score_archive(**completed)


def test_capture_hash_guard_precedes_label_access(completed, monkeypatch):
    path = attempt(completed) / "transcript.jsonl"
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b"\n")
    deny_labels(monkeypatch)
    with pytest.raises(ValueError, match="SHA256"):
        archive.score_archive(**completed)


def test_recomputed_capture_hash_does_not_authorize_changed_launch(completed, monkeypatch):
    path = attempt(completed) / "started.json"
    changed = load(path)
    changed["command"]["stdin"] += " changed request"
    rewrite(path, changed)
    rehash(completed)
    deny_labels(monkeypatch)
    with pytest.raises(ValueError, match="launch"):
        archive.score_archive(**completed)


def test_stored_decision_replay_rejected(completed, monkeypatch):
    path = attempt(completed) / "result.json"
    changed = load(path)
    changed["run"]["decision"]["ranking"].reverse()
    changed["run"]["decision"]["choice"] = changed["run"]["decision"]["ranking"][0]
    rewrite(path, changed)
    deny_labels(monkeypatch)
    with pytest.raises(ValueError, match="captured transcript"):
        archive.score_archive(**completed)


def test_other_arm_captures_cannot_be_replayed(completed, monkeypatch):
    source_attempt, target = attempt(completed), attempt(completed, "rpm_wm")
    for name in runner.CAPTURE_FILES | {"result.json"}:
        (target / name).chmod(0o600)
        shutil.copyfile(source_attempt / name, target / name)
    deny_labels(monkeypatch)
    with pytest.raises(ValueError, match="different evidence"):
        archive.score_archive(**completed)


def test_bundle_or_study_mismatch_precedes_labels(completed, monkeypatch):
    completed["bundle_sha256"] = "0" * 64
    deny_labels(monkeypatch)
    with pytest.raises(ValueError):
        archive.score_archive(**completed)


def test_running_execution_lock_blocks_label_access(completed, monkeypatch):
    deny_labels(monkeypatch)
    with archive._execution_lock(completed["execution_dir"]), pytest.raises(BlockingIOError):
        archive.score_archive(**completed)


def test_timeout_is_completed_but_remains_failed_no_imputation(completed):
    directory = attempt(completed)
    (directory / "transcript.jsonl").chmod(0o600)
    (directory / "transcript.jsonl").write_bytes(b"")
    capture = {"returncode": -15, "elapsed_seconds": 30, "timed_out": True, "runner_failure": None}
    rewrite(directory / "capture.json", capture)
    bundle = runner.validate_bundle(
        completed["bundle_path"], expected_sha256=completed["bundle_sha256"]
    )
    normalized = runner._normalized(directory, bundle, "case-one", "rpm", capture)
    result = load(directory / "result.json")
    result["run"] = normalized
    rewrite(directory / "result.json", result)
    rehash(completed)
    summary = archive.score_archive(**completed)
    report = load(completed["output_dir"] / "report.json")
    assert summary["completed_attempts"] == 2 and summary["paired_valid_cases"] == 0
    assert report["status"] == "incomplete_comparison"
    assert report["cost"]["missing_attempts"] == 0
    assert report["cost"]["attempts_with_unknown_cost"] == 1
    assert "selected_model_accuracy" not in report["cases"][0]["arms"]["rpm"]


def test_positive_inventory_projection_never_decodes_plan_bodies(completed):
    path = completed["inventory_path"]
    path.write_text(path.read_text().replace('{"forbidden_to_decode": true}', "INVALID_BODY"))
    completed["inventory_sha256"] = runner.sha(path.read_bytes())
    assert archive.score_archive(**completed)["paired_valid_cases"] == 1


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonofficial", "changed_pin"])
def test_exact_official_candidate_label_join_refuses_gaps(completed, mutation):
    path = completed["inventory_path"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "nonofficial":
        rows[0]["label"]["official_metric"] = None
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    if mutation != "changed_pin":
        completed["inventory_sha256"] = runner.sha(path.read_bytes())
    else:
        completed["inventory_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        archive.score_archive(**completed)
    assert not completed["output_dir"].exists()


def test_existing_output_refused_before_label_access(completed, monkeypatch):
    completed["output_dir"].mkdir()
    deny_labels(monkeypatch)
    with pytest.raises(FileExistsError):
        archive.score_archive(**completed)
