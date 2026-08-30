import json
import subprocess
from pathlib import Path

import pytest

from awm import paths
from awm import ptb_experiments as ptb

MANIFEST = paths.REPO_ROOT / "experiments/posttrainbench/gsm8k-claude5-1m-batch1.yaml"


def test_manifest_is_exact_approved_matrix() -> None:
    data = ptb.load_manifest(MANIFEST)
    launches = ptb.build_launches(data)
    assert [launch.cell_id for launch in launches] == ["b1", "b2", "b3", "b4", "b5", "b6"]
    assert all("official" in launch.command for launch in launches)
    assert all(
        launch.environment["POST_TRAIN_BENCH_REQUIRE_COMPLETE"] == "1" for launch in launches
    )
    assert all(
        launch.environment["POST_TRAIN_BENCH_SKIP_CLI_UPDATE"] == "1" for launch in launches
    )
    assert all(
        launch.environment["POST_TRAIN_BENCH_EVALUATION_CONTAINER_SHA256"]
        == "72748f77f9fe5a1abe925bb532c1da64d80b1dcce7849179c9546700099448f8"
        for launch in launches
    )
    assert launches[0].environment["POST_TRAIN_BENCH_BASE_MODEL_REVISION"] == (
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
    )
    assert launches[1].environment["POST_TRAIN_BENCH_BASE_MODEL_REVISION"] == (
        "906bfd4b4dc7f14ee4320094d8b41684abff8539"
    )
    assert launches[2].environment["POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD"].endswith(
        "claude-opus-5-1m-max.json"
    )
    assert launches[5].environment["POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD"].endswith(
        "claude-opus-5-1m-xhigh.json"
    )
    held = ptb.build_launches(data, hold=True)
    assert all("--hold" in launch.command for launch in held)


def test_pilot_is_b6_shape_and_one_hour() -> None:
    (launch,) = ptb.build_launches(ptb.load_manifest(MANIFEST), pilot=True)
    assert launch.cell_id == "b6"
    assert launch.command[launch.command.index("--hours") + 1] == "1"
    assert any("pilot_1h" in argument for argument in launch.command)
    assert "--hold" not in launch.command


def test_manifest_rejects_non_1m_contract() -> None:
    data = ptb.load_manifest(MANIFEST)
    data["contract"]["context_tokens"] = 200_000
    with pytest.raises(ptb.ExperimentError, match="context_tokens"):
        ptb.validate_manifest(data)


def test_manifest_pins_all_runtime_images() -> None:
    contract = ptb.load_manifest(MANIFEST)["contract"]
    assert contract["container"]["sha256"]
    assert contract["evaluation_container"] == {
        "name": "vllm_debug.sif",
        "sha256": "72748f77f9fe5a1abe925bb532c1da64d80b1dcce7849179c9546700099448f8",
    }
    assert contract["official_judge_container_sha256"]
    assert contract["base_models"] == {
        "google/gemma-3-4b-pt": {
            "revision": "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
        },
        "Qwen/Qwen3-4B-Base": {
            "revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539"
        },
    }
    assert contract["agent_cli_version"] == "2.1.219"
    assert contract["agent_auth"] == {
        "provider": "vertex",
        "project": "sercan-v1",
        "region": "global",
    }


def test_result_audit_requires_full_official_flow(tmp_path: Path) -> None:
    issues = ptb.audit_result(tmp_path)
    assert "missing or empty: metrics.json" in issues
    assert "missing or empty: judgement_general.json" in issues


def test_receipt_validation(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"schema_version": 1, "jobs": [{"cell_id": "b6", "job_id": "1"}]}')
    assert ptb.load_receipt(receipt)["jobs"][0]["cell_id"] == "b6"


def test_formal_submit_holds_all_jobs_before_one_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = ptb.load_manifest(MANIFEST)
    fake_launches = [
        ptb.Launch(
            cell_id=f"b{index}",
            command=("fake-submit", f"b{index}", "--hold"),
            environment={
                "POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD": f"/evidence/b{index}.json",
                "POST_TRAIN_BENCH_CONTEXT_VALIDATION_SHA256": f"{index:064x}",
            },
        )
        for index in range(1, 7)
    ]
    monkeypatch.setattr(ptb, "local_issues", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(ptb, "site_issues", list)
    monkeypatch.setattr(
        ptb,
        "source_snapshot",
        lambda: {
            "top_commit": "1" * 40,
            "ptb_commit": "2" * 40,
            "top_status": "",
            "ptb_status": "",
        },
    )
    monkeypatch.setattr(ptb, "dry_run", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(ptb, "read_ptb_env", dict)
    monkeypatch.setattr(ptb.paths, "data_root", lambda: tmp_path)
    monkeypatch.setattr(ptb, "build_launches", lambda *_args, **_kwargs: fake_launches)

    commands: list[tuple[str, ...]] = []

    def fake_run(command: list[str] | tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        normalized = tuple(command)
        commands.append(normalized)
        if normalized[:2] == ("scontrol", "release"):
            return subprocess.CompletedProcess(command, 0, "", "")
        job_id = 9000 + len([item for item in commands if item[0] == "fake-submit"])
        return subprocess.CompletedProcess(command, 0, f"Submitted Slurm job {job_id}\n", "")

    monkeypatch.setattr(ptb.subprocess, "run", fake_run)
    receipt_path = ptb.submit(data)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    submitted = [command for command in commands if command[0] == "fake-submit"]
    assert len(submitted) == 6
    assert all("--hold" in command for command in submitted)
    assert commands[-1] == ("scontrol", "release", "9001,9002,9003,9004,9005,9006")
    assert receipt["state"] == "submitted"
    assert len(receipt["jobs"]) == 6
    assert set(receipt["context_validation"]) == {f"b{index}" for index in range(1, 7)}
