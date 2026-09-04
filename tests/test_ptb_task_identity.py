"""Independent task identity at the AWM→PTB boundary; synthetic files only."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from awm import ptb_experiments as ptb
from awm import ptb_ops, ptb_results, slurm_queue


def provenance(path, task):
    path.mkdir(parents=True, exist_ok=True)
    (path / "runtime_provenance.json").write_text(json.dumps({"experiment": {"task": task}}))


@pytest.mark.parametrize("actual", [None, "gsm8k", "", True])
def test_humaneval_cannot_select_legacy_validation_by_claim(actual, tmp_path, monkeypatch):
    provenance(tmp_path, actual)
    calls = []

    def old_validator(command, **kwargs):
        calls.append(command)
        # A legacy CLI refuses the new flag. It must not be retried without it.
        return subprocess.CompletedProcess(command, 2, "", "unrecognized arguments: --expected-task")

    monkeypatch.setattr(ptb.subprocess, "run", old_validator)
    issues = ptb.audit_result(tmp_path, expected_task="humaneval")
    assert "runtime provenance task differs from independent expected task" in issues
    assert len(calls) == 1 and calls[0][-2:] == ["--expected-task", "humaneval"]


def test_old_gsm_validator_retains_cli_but_requires_correct_identity(tmp_path, monkeypatch):
    provenance(tmp_path, "gsm8k")
    calls = []

    def old_validator(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(ptb.subprocess, "run", old_validator)
    assert ptb.audit_result(tmp_path, expected_task="gsm8k") == []
    assert "--expected-task" not in calls[-1]
    assert ptb.audit_result(tmp_path) == ["independent expected task is required for completion validation"]
    provenance(tmp_path, None)
    assert ptb.audit_result(tmp_path, expected_task="gsm8k")


def test_new_humaneval_validator_receives_authoritative_task(tmp_path, monkeypatch):
    provenance(tmp_path, "humaneval")

    def validator(command, **kwargs):
        assert command[-2:] == ["--expected-task", "humaneval"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(ptb.subprocess, "run", validator)
    assert ptb.audit_result(tmp_path, expected_task="humaneval") == []


def receipt(task="humaneval"):
    return {"schema_version": 1, "batch_id": "invented-batch", "contract": {"task": task},
            "jobs": [{"job_id": "101", "cell_id": "h01"}], "cells": [{"id": "h01"}]}


@pytest.mark.parametrize("change", ["job", "cell", "duplicate", "missing", "bad-record"])
def test_receipt_task_requires_unique_job_cell(change):
    value = receipt()
    if change == "job":
        value["jobs"][0]["job_id"] = "102"
    elif change == "cell":
        value["cells"][0]["id"] = "other"
    elif change == "duplicate":
        value["jobs"].append(value["jobs"][0])
    elif change == "missing":
        value.pop("cells")
    else:
        value["jobs"] = ["invalid record"]
    with pytest.raises(ptb.ExperimentError):
        ptb.receipt_task(value, "101", "h01")


def test_result_discovery_passes_manifest_task_even_when_result_claims_other(tmp_path, monkeypatch):
    root = tmp_path / "agent" / "result"
    root.mkdir(parents=True)
    (root / "runtime_provenance.json").write_text(json.dumps({
        "experiment": {"batch_id": "invented-batch", "cell_id": "h01", "task": "gsm8k"}}))
    observed = []

    def audit(path, *, expected_task):
        observed.append(expected_task)
        return ["intentionally incomplete"]

    monkeypatch.setattr(ptb_results, "_results_root", lambda: tmp_path)
    monkeypatch.setattr(ptb_results, "_receipts_root", lambda: tmp_path / "receipts")
    monkeypatch.setattr(ptb_results.ptb, "audit_result", audit)
    attempts = ptb_results.discover_attempts({"batch_id": "invented-batch",
        "contract": {"task": "humaneval"}, "cells": [{"id": "h01"}]})
    assert observed == ["humaneval"] and not attempts["h01"][0]["complete"]


def test_queue_task_comes_from_receipt_not_result(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt()))
    source, job = {"path": str(path)}, {"job_id": "101", "cell_id": "h01"}
    assert slurm_queue._source_task(source, job) == "humaneval"
    assert slurm_queue._source_task(source, {**job, "job_id": "999"}) is None
    assert not slurm_queue._has_validated_ptb_result("101", expected_task=None)


def test_harvest_keeps_artifacts_without_asserting_unscoped_completion(tmp_path, monkeypatch):
    result = tmp_path / "source"
    provenance(result, "gsm8k")
    (result / "metrics.json").write_text('{"accuracy": 0.5}')
    seen = []

    def audit(path, *, expected_task):
        seen.append(expected_task)
        return ["missing independent identity"] if expected_task is None else []

    monkeypatch.setattr(ptb_ops, "audit_result", audit)
    output = tmp_path / "bundle"
    status = ptb_ops.harvest_job(result, output, batch="invented", cell="h01", job_id="101")
    assert seen == [None] and not status["complete"] and not status["eligible"]
    assert (output / "metrics.json").is_file()


@pytest.mark.parametrize("copies, expected", [(0, None), (1, "humaneval"), (2, None)])
def test_manual_harvest_requires_one_exact_receipt(tmp_path, monkeypatch, copies, expected):
    directory = tmp_path / "results/ptb/invented-batch"
    directory.mkdir(parents=True)
    for index in range(copies):
        (directory / f"formal-{index}.json").write_text(json.dumps(receipt()))
    calls = []

    def harvest(*args, **kwargs):
        calls.append(kwargs)
        return {"complete": False, "eligible": False, "quarantined": False, "skipped": []}

    monkeypatch.setattr(ptb_ops.paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ptb_ops, "harvest_job", harvest)
    args = SimpleNamespace(batch="invented-batch", cell="h01", job="101", job_name=None,
                           state="FAILED", result_dir=None, out=None)
    assert ptb_ops.harvest_cli(args) == 1
    assert calls[0]["expected_task"] == expected


def test_reconcile_harvest_passes_receipt_task(tmp_path, monkeypatch):
    directory = tmp_path / "results/ptb/invented-batch"
    directory.mkdir(parents=True)
    (directory / "formal.json").write_text(json.dumps(receipt()))
    calls = []

    def harvest(*args, **kwargs):
        calls.append(kwargs)
        return {"complete": False, "eligible": False, "quarantined": False, "issues": [],
                "judge_flags": [], "accuracy": None}

    monkeypatch.setattr(ptb_ops, "result_for_job", lambda _: None)
    monkeypatch.setattr(ptb_ops, "harvest_job", harvest)
    action = ptb_ops.Action("harvest", "invented-batch", "test only", cell="h01",
                           job_id="101", receipt="formal.json", state="FAILED")
    written = ptb_ops.apply([action], tmp_path)
    assert calls[0]["expected_task"] == "humaneval"
    assert "judges-unverified" in written[0] and not written[0].endswith(" clean")
