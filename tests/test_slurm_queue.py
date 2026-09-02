import json
from pathlib import Path

from awm import slurm_queue


def test_register_receipt_freezes_job_ids_and_names(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "batch_id": "batch-v1",
                "jobs": [
                    {"cell_id": "a1", "job_id": "101", "job_name": "branch.batch.a1"},
                    {"cell_id": "a2", "job_id": "102", "job_name": "branch.batch.a2"},
                ],
            }
        )
    )
    registry = tmp_path / "registry.json"

    assert slurm_queue.register_receipt(receipt, registry_path=registry) == registry
    data = json.loads(registry.read_text(encoding="utf-8"))

    assert data["sources"][0]["label"] == "batch-v1"
    assert data["sources"][0]["jobs"] == [
        {"cell_id": "a1", "job_id": "101", "job_name": "branch.batch.a1"},
        {"cell_id": "a2", "job_id": "102", "job_name": "branch.batch.a2"},
    ]

    assert slurm_queue.unregister_receipt(receipt, registry_path=registry) == registry
    assert json.loads(registry.read_text(encoding="utf-8"))["sources"] == []


def test_snapshot_flags_unregistered_jobs_on_owned_nodes(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "registry.json"
    data = slurm_queue._default_registry()
    data["sources"] = [
        {
            "id": "receipt:test",
            "kind": "receipt",
            "label": "test batch",
            "jobs": [{"cell_id": "a1", "job_id": "101", "job_name": "branch.batch.a1"}],
        }
    ]
    registry.write_text(json.dumps(data), encoding="utf-8")

    def fake_command(command: list[str]) -> str:
        if command[0] == "squeue":
            return (
                "101|RUNNING|slurm2-a3nodesetondem-0|slurm2-a3nodesetondem-0|"
                "branch.batch.a1|00:10|root|ptb-a3\n"
                "999|RUNNING|slurm2-a3nodesetondem-1|slurm2-a3nodesetondem-1|"
                "unregistered|00:05|root|ptb-a3\n"
            )
        if command[:3] == ["scontrol", "show", "node"]:
            node = command[3]
            return f"NodeName={node} State=MIXED AllocTRES=cpu=16,gres/gpu=1\n"
        raise AssertionError(command)

    monkeypatch.setattr(slurm_queue, "_command", fake_command)
    snapshot = slurm_queue.collect_snapshot(registry)

    assert snapshot["ownership_ok"] is False
    assert snapshot["queue_name"] == "gangda"
    assert [job["job_id"] for job in snapshot["unknown_jobs"]] == ["999"]
    assert snapshot["sources"][0]["counts"] == {"RUNNING": 1}
    rendered = slurm_queue.render_snapshot(snapshot)
    assert "QUEUE gangda" in rendered
    assert "OWNERSHIP FAIL" in rendered


def test_snapshot_accepts_only_exact_registered_names(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "registry.json"
    data = slurm_queue._default_registry()
    data["sources"] = [
        {
            "id": "job:test",
            "kind": "explicit_jobs",
            "label": "recovery",
            "jobs": [{"job_id": "101", "job_name": "expected"}],
        }
    ]
    registry.write_text(json.dumps(data), encoding="utf-8")

    def fake_command(command: list[str]) -> str:
        if command[0] == "squeue":
            return (
                "101|RUNNING|slurm2-a3nodesetondem-0|slurm2-a3nodesetondem-0|"
                "wrong|00:10|root|ptb-a3\n"
            )
        if command[:3] == ["scontrol", "show", "node"]:
            return f"NodeName={command[3]} State=MIXED AllocTRES=gres/gpu=1\n"
        raise AssertionError(command)

    monkeypatch.setattr(slurm_queue, "_command", fake_command)
    snapshot = slurm_queue.collect_snapshot(registry)

    assert snapshot["ownership_ok"] is False
    assert snapshot["name_mismatches"] == [
        {"job_id": "101", "expected": "expected", "actual": "wrong"}
    ]


def test_unknown_jobs_become_enforceable_only_after_grace() -> None:
    snapshot = {
        "unknown_jobs": [{"job_id": "999"}],
        "name_mismatches": [{"job_id": "101"}],
    }

    seen, due = slurm_queue._enforcement_due(snapshot, {}, now=100.0, grace=60)
    assert seen == {"101": 100.0, "999": 100.0}
    assert due == []

    seen, due = slurm_queue._enforcement_due(snapshot, seen, now=161.0, grace=60)
    assert seen == {"101": 100.0, "999": 100.0}
    assert due == ["101", "999"]


def test_current_failures_and_history_are_separate_views(monkeypatch) -> None:
    monkeypatch.setattr(slurm_queue, "_has_validated_ptb_result", lambda _job_id: False)
    snapshot = {
        "updated_at": "2026-09-01T00:00:00+00:00",
        "queue_name": "gangda",
        "owner": "owner",
        "ownership_ok": True,
        "unknown_jobs": [],
        "name_mismatches": [],
        "nodes": [],
        "gpus_allocated": 1,
        "gpus_total": 1,
        "sources": [
            {
                "id": "old",
                "label": "old launch",
                "kind": "receipt",
                "path": "/tmp/old.json",
                "batch_id": "batch",
                "registered_at": "2026-09-01T00:00:00+00:00",
                "manifest": "/tmp/manifest.yaml",
                "spec": "doc/spec.md",
                "counts": {"FAILED": 1},
                "active": 0,
                "jobs": [
                    {
                        "job_id": "101",
                        "cell_id": "g01r1",
                        "state": "FAILED",
                        "elapsed": "00:01",
                        "nodes": "node0",
                    }
                ],
            },
            {
                "id": "retry",
                "label": "retry launch",
                "kind": "receipt",
                "path": "/tmp/retry.json",
                "batch_id": "batch",
                "registered_at": "2026-09-01T01:00:00+00:00",
                "manifest": "/tmp/manifest.yaml",
                "spec": "doc/spec.md",
                "counts": {"RUNNING": 1},
                "active": 1,
                "jobs": [
                    {
                        "job_id": "102",
                        "cell_id": "g01r1",
                        "state": "RUNNING",
                        "elapsed": "00:10",
                        "nodes": "node0",
                    }
                ],
            },
        ],
    }

    current = slurm_queue.render_snapshot(snapshot)
    assert "retry launch" in current
    assert "old launch" not in current
    assert slurm_queue.failure_records(snapshot) == []
    assert "NO UNRESOLVED FAILURES" in slurm_queue.render_failures(snapshot)
    assert "old launch" in slurm_queue.render_history(snapshot)
    assert "resolved_by=102:RUNNING" in slurm_queue.render_failures(snapshot, include_resolved=True)


def test_validated_result_resolves_scheduler_failure(monkeypatch) -> None:
    monkeypatch.setattr(slurm_queue, "_has_validated_ptb_result", lambda job_id: job_id == "101")
    snapshot = {
        "updated_at": "2026-09-01T00:00:00+00:00",
        "queue_name": "gangda",
        "unknown_jobs": [],
        "name_mismatches": [],
        "sources": [
            {
                "id": "recovery",
                "label": "recovery",
                "batch_id": "",
                "registered_at": "2026-09-01T00:00:00+00:00",
                "jobs": [{"job_id": "101", "state": "FAILED", "cell_id": ""}],
            }
        ],
    }

    assert slurm_queue.failure_records(snapshot) == []
    resolved = slurm_queue.failure_records(snapshot, include_resolved=True)
    assert resolved[0]["replacement"] == {
        "source": "validated PTB result",
        "job_id": "101",
        "state": "COMPLETE",
    }


def test_show_resolves_job_to_receipt_cell(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "source": {"top_commit": "a" * 40, "ptb_commit": "b" * 40},
                "cells": [
                    {
                        "id": "g01r1",
                        "task": "gsm8k",
                        "base_model": "Qwen/Qwen3-1.7B-Base",
                        "agent": "claude_vertex_max",
                        "effort": "max",
                        "context_tokens": 1_000_000,
                        "replicate": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    snapshot = {
        "queue_name": "gangda",
        "sources": [
            {
                "label": "batch",
                "kind": "receipt",
                "path": str(receipt),
                "batch_id": "batch-v1",
                "manifest": "/tmp/manifest.yaml",
                "spec": "doc/spec.md",
                "jobs": [
                    {
                        "job_id": "101",
                        "cell_id": "g01r1",
                        "state": "RUNNING",
                        "nodes": "node0",
                        "elapsed": "00:10",
                        "expected_name": "branch.batch.g01r1",
                    }
                ],
            }
        ],
    }

    explanation = slurm_queue.explain_job(snapshot, "101")

    assert explanation["cell"]["task"] == "gsm8k"
    assert explanation["frozen_source"]["top_commit"] == "a" * 40
    rendered = slurm_queue.render_job_explanation(explanation)
    assert "base_model=Qwen/Qwen3-1.7B-Base" in rendered
    assert "manifest=/tmp/manifest.yaml" in rendered


def test_default_registry_splits_gangda_into_two_fixed_sixteen_gpu_subqueues() -> None:
    registry = slurm_queue._default_registry()

    assert registry["subqueues"] == {
        "gangda_exp-protocol-evolve": {
            "branches": ["gangda_exp_protocol_evolve"],
            "gpu_limit": 16,
            "nodes": ["slurm2-a3nodesetondem-0", "slurm2-a3nodesetondem-1"],
        },
        "gangda_wma_evolve": {
            "branches": ["gangda_wma_evolve"],
            "gpu_limit": 16,
            "nodes": ["slurm2-a3nodesetondem-2", "slurm2-a3nodesetondem-3"],
        },
    }


def test_register_receipt_assigns_the_branch_subqueue(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "batch_id": "ep-r00",
                "ownership": {"branch": "gangda_exp_protocol_evolve"},
                "jobs": [
                    {
                        "cell_id": "p01r1",
                        "job_id": "101",
                        "job_name": "gangda_exp_protocol_evolve.ptb.ep-r00.p01r1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"

    slurm_queue.register_receipt(receipt, registry_path=registry)

    data = json.loads(registry.read_text(encoding="utf-8"))
    assert data["sources"][0]["subqueue"] == "gangda_exp-protocol-evolve"


def test_snapshot_renders_each_subqueue_capacity(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "registry.json"
    data = slurm_queue._default_registry()
    data["sources"] = [
        {
            "id": "receipt:exp",
            "kind": "receipt",
            "label": "exp round",
            "subqueue": "gangda_exp-protocol-evolve",
            "jobs": [
                {
                    "cell_id": "p01r1",
                    "job_id": "101",
                    "job_name": "gangda_exp_protocol_evolve.ptb.ep-r00.p01r1",
                }
            ],
        }
    ]
    registry.write_text(json.dumps(data), encoding="utf-8")

    def fake_command(command: list[str]) -> str:
        if command[0] == "squeue":
            return (
                "101|RUNNING|slurm2-a3nodesetondem-0|slurm2-a3nodesetondem-0|"
                "gangda_exp_protocol_evolve.ptb.ep-r00.p01r1|00:10|root|ptb-a3\n"
            )
        if command[:3] == ["scontrol", "show", "node"]:
            allocated = 1 if command[3].endswith("-0") else 0
            return f"NodeName={command[3]} State=MIXED AllocTRES=gres/gpu={allocated}\n"
        raise AssertionError(command)

    monkeypatch.setattr(slurm_queue, "_command", fake_command)
    snapshot = slurm_queue.collect_snapshot(registry)

    assert [(item["name"], item["gpus_allocated"], item["gpus_total"]) for item in snapshot["subqueues"]] == [
        ("gangda_exp-protocol-evolve", 1, 16),
        ("gangda_wma_evolve", 0, 16),
    ]
    rendered = slurm_queue.render_snapshot(snapshot)
    assert "gangda_exp-protocol-evolve: GPUS 1/16" in rendered
    assert "gangda_wma_evolve: GPUS 0/16" in rendered

    exp_only = slurm_queue.render_snapshot(
        snapshot, subqueue="gangda_exp-protocol-evolve"
    )
    assert "SUBQUEUE gangda_exp-protocol-evolve" in exp_only
    assert "exp round" in exp_only
    assert "gangda_wma_evolve" not in exp_only


def test_snapshot_fails_for_registered_jobs_outside_nodes_or_over_gpu_limit(
    tmp_path: Path, monkeypatch
) -> None:
    registry = tmp_path / "registry.json"
    data = slurm_queue._default_registry()
    jobs = [
        {
            "cell_id": f"p01r{index:02d}",
            "job_id": str(100 + index),
            "job_name": f"gangda_exp_protocol_evolve.ptb.batch.p01r{index:02d}",
        }
        for index in range(1, 18)
    ]
    data["sources"] = [
        {
            "id": "receipt:exp",
            "kind": "receipt",
            "label": "overflow batch",
            "subqueue": "gangda_exp-protocol-evolve",
            "jobs": jobs,
        }
    ]
    registry.write_text(json.dumps(data), encoding="utf-8")

    def fake_command(command: list[str]) -> str:
        if command[0] == "squeue":
            rows = []
            for index, job in enumerate(jobs):
                node = (
                    "slurm2-a3nodeset-9"
                    if index == 16
                    else f"slurm2-a3nodesetondem-{index % 2}"
                )
                rows.append(
                    f"{job['job_id']}|RUNNING|{node}|{node}|{job['job_name']}|"
                    "00:10|root|ptb-a3|gres/gpu:1"
                )
            return "\n".join(rows) + "\n"
        if command[:3] == ["scontrol", "show", "node"]:
            return f"NodeName={command[3]} State=MIXED AllocTRES=gres/gpu=8\n"
        raise AssertionError(command)

    monkeypatch.setattr(slurm_queue, "_command", fake_command)
    snapshot = slurm_queue.collect_snapshot(registry)

    assert snapshot["ownership_ok"] is False
    assert snapshot["placement_violations"] == [
        {
            "job_id": "117",
            "subqueue": "gangda_exp-protocol-evolve",
            "expected_nodes": [
                "slurm2-a3nodesetondem-0",
                "slurm2-a3nodesetondem-1",
            ],
            "actual_nodes": ["slurm2-a3nodeset-9"],
            "outside_nodes": ["slurm2-a3nodeset-9"],
        }
    ]
    assert snapshot["capacity_violations"] == [
        {
            "subqueue": "gangda_exp-protocol-evolve",
            "registered_running_gpus": 17,
            "gpu_limit": 16,
        }
    ]
    rendered = slurm_queue.render_snapshot(snapshot)
    assert "OWNERSHIP FAIL" in rendered
    assert "PLACEMENT VIOLATIONS" in rendered
    assert "CAPACITY VIOLATIONS" in rendered
