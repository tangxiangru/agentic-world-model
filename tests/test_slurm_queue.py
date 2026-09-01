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
    assert [job["job_id"] for job in snapshot["unknown_jobs"]] == ["999"]
    assert snapshot["sources"][0]["counts"] == {"RUNNING": 1}
    assert "OWNERSHIP FAIL" in slurm_queue.render_snapshot(snapshot)


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
