import json
import subprocess
from pathlib import Path

import pytest

from awm import paths
from awm import ptb_experiments as ptb

MANIFEST = paths.REPO_ROOT / "experiments/posttrainbench/gsm8k-opus5-4x4-batch1.yaml"
DUAL_MANIFEST = (
    paths.REPO_ROOT / "experiments/posttrainbench/gsm8k-aime2025-opus5-4x4x2-batch1.yaml"
)


def test_manifest_is_exact_approved_matrix() -> None:
    data = ptb.load_manifest(MANIFEST)
    assert data["ownership"] == {
        "branch": "gangda_trial_0828",
        "spec": "doc/spec/2026-08-30-ptb-gpu-slicing-and-gsm8k-batch1.md",
    }
    launches = ptb.build_launches(data)
    assert [launch.cell_id for launch in launches] == [f"b{index:02d}" for index in range(1, 17)]
    assert all("official" in launch.command for launch in launches)
    assert all(
        launch.environment["POST_TRAIN_BENCH_REQUIRE_COMPLETE"] == "1" for launch in launches
    )
    assert all(launch.environment["POST_TRAIN_BENCH_SKIP_CLI_UPDATE"] == "1" for launch in launches)
    assert all(
        launch.environment["HF_HOME"] == str(paths.REPO_ROOT / "data/ptb/hf") for launch in launches
    )
    assert all(
        launch.command[launch.command.index("--run-branch") + 1] == "gangda_trial_0828"
        for launch in launches
    )
    assert [launch.command[launch.command.index("--job-name") + 1] for launch in launches] == [
        f"gangda_trial_0828.ptb.gsm8k-opus5-4x4-batch1.b{index:02d}.formal.r1"
        for index in range(1, 17)
    ]
    assert [
        launch.command[launch.command.index("--experiment-name") + 1] for launch in launches
    ] == [
        f"_gangda_trial_0828_gsm8k-opus5-4x4-batch1_b{index:02d}_formal_r1"
        for index in range(1, 17)
    ]
    assert all(
        launch.environment["POST_TRAIN_BENCH_EVALUATION_CONTAINER_SHA256"]
        == "72748f77f9fe5a1abe925bb532c1da64d80b1dcce7849179c9546700099448f8"
        for launch in launches
    )
    assert launches[0].environment["POST_TRAIN_BENCH_BASE_MODEL_REVISION"] == (
        "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
    )
    assert launches[1].environment["POST_TRAIN_BENCH_BASE_MODEL_REVISION"] == (
        "906bfd4b4dc7f14ee4320094d8b41684abff8539"
    )
    assert launches[2].environment["POST_TRAIN_BENCH_BASE_MODEL_REVISION"] == (
        "d78a42f79198603e614095753484a04c10c2b940"
    )
    assert launches[3].environment["POST_TRAIN_BENCH_BASE_MODEL_REVISION"] == (
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
    )
    assert (
        launches[0]
        .environment["POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD"]
        .endswith("claude-opus-5-1m-max.json")
    )
    assert (
        launches[4]
        .environment["POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD"]
        .endswith("claude-opus-5-1m-xhigh.json")
    )
    assert (
        launches[8]
        .environment["POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD"]
        .endswith("claude-opus-5-1m-high.json")
    )
    assert (
        launches[12]
        .environment["POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD"]
        .endswith("claude-opus-5-200k-max.json")
    )
    assert all(
        launch.environment["POST_TRAIN_BENCH_EXPECTED_CONTEXT_TOKENS"] == "1000000"
        for launch in launches[:12]
    )
    assert all(
        launch.environment["POST_TRAIN_BENCH_EXPECTED_CONTEXT_TOKENS"] == "200000"
        for launch in launches[12:]
    )
    held = ptb.build_launches(data, hold=True)
    assert all("--hold" in launch.command for launch in held)


def test_dual_task_manifest_is_one_atomic_thirty_two_cell_matrix() -> None:
    data = ptb.load_manifest(DUAL_MANIFEST)
    launches = ptb.build_launches(data)

    assert data["contract"]["tasks"] == ["gsm8k", "aime2025"]
    assert [launch.cell_id for launch in launches] == [
        *(f"g{index:02d}" for index in range(1, 17)),
        *(f"a{index:02d}" for index in range(1, 17)),
    ]
    tasks = [launch.command[launch.command.index("--eval") + 1] for launch in launches]
    assert tasks == ["gsm8k"] * 16 + ["aime2025"] * 16
    assert all("--hold" in launch.command for launch in ptb.build_launches(data, hold=True))


def test_dual_task_pilot_covers_both_evaluation_paths() -> None:
    launches = ptb.build_launches(ptb.load_manifest(DUAL_MANIFEST), pilot=True)

    assert [launch.cell_id for launch in launches] == ["g06", "a06"]
    assert [launch.command[launch.command.index("--eval") + 1] for launch in launches] == [
        "gsm8k",
        "aime2025",
    ]
    assert all(launch.command[launch.command.index("--hours") + 1] == "1" for launch in launches)
    assert all("--hold" not in launch.command for launch in launches)


def test_aime2025_decontamination_asset_is_the_complete_numeric_test_set() -> None:
    task = ptb.PTB_ROOT / "src/eval/tasks/aime2025"
    test_rows = json.loads((task / "test_data.json").read_text(encoding="utf-8"))

    assert len(test_rows) == 30
    assert all(row["question"] for row in test_rows)
    assert all(str(row["answer"]).isdigit() for row in test_rows)


def test_frozen_source_tracks_both_task_decontamination_assets() -> None:
    assert ptb._is_git_tracked(ptb.PTB_ROOT, "src/eval/tasks/gsm8k/test_data.json")
    assert ptb._is_git_tracked(ptb.PTB_ROOT, "src/eval/tasks/aime2025/test_data.json")


def test_pilot_is_b06_shape_and_one_hour() -> None:
    (launch,) = ptb.build_launches(ptb.load_manifest(MANIFEST), pilot=True)
    assert launch.cell_id == "b06"
    assert launch.command[launch.command.index("--hours") + 1] == "1"
    assert any("pilot-1h" in argument for argument in launch.command)
    assert (
        launch.command[launch.command.index("--job-name") + 1]
        == "gangda_trial_0828.ptb.gsm8k-opus5-4x4-batch1.b06.pilot-1h.r1"
    )
    assert "--hold" not in launch.command


def test_manifest_rejects_wrong_context_setup() -> None:
    data = ptb.load_manifest(MANIFEST)
    data["cells"][0]["context_tokens"] = 200_000
    with pytest.raises(ptb.ExperimentError, match="4x4"):
        ptb.validate_manifest(data)


def test_source_ownership_rejects_wrong_branch() -> None:
    data = ptb.load_manifest(MANIFEST)
    with pytest.raises(ptb.ExperimentError, match="does not match"):
        ptb.assert_source_ownership(data, {"top_branch": "someone_else"})


def test_manifest_pins_all_runtime_images() -> None:
    contract = ptb.load_manifest(MANIFEST)["contract"]
    assert contract["container"]["sha256"]
    assert contract["evaluation_container"] == {
        "name": "vllm_debug.sif",
        "sha256": "72748f77f9fe5a1abe925bb532c1da64d80b1dcce7849179c9546700099448f8",
    }
    assert contract["official_judge_container_sha256"]
    assert contract["base_models"] == {
        "Qwen/Qwen3-1.7B-Base": {"revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"},
        "Qwen/Qwen3-4B-Base": {"revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539"},
        "HuggingFaceTB/SmolLM3-3B-Base": {"revision": "d78a42f79198603e614095753484a04c10c2b940"},
        "google/gemma-3-4b-pt": {"revision": "cc012e0a6d0787b4adcc0fa2c4da74402494554d"},
    }
    assert contract["agent_cli_version"] == "2.1.219"
    assert contract["agent_auth"] == {
        "provider": "vertex",
        "project": "sercan-v1",
        "region": "global",
    }


def test_base_model_snapshot_accepts_monolithic_safetensors(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    assert ptb._base_model_snapshot_issues("Qwen/example", "a" * 40, tmp_path) == []


def test_result_audit_requires_full_official_flow(tmp_path: Path) -> None:
    issues = ptb.audit_result(tmp_path)
    assert "missing or empty: metrics.json" in issues
    assert "missing or empty: judgement_general.json" in issues


def test_receipt_validation(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"schema_version": 1, "jobs": [{"cell_id": "b06", "job_id": "1"}]}')
    assert ptb.load_receipt(receipt)["jobs"][0]["cell_id"] == "b06"


def test_formal_submit_holds_all_jobs_before_one_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = ptb.load_manifest(DUAL_MANIFEST)
    cell_ids = [
        *(f"g{index:02d}" for index in range(1, 17)),
        *(f"a{index:02d}" for index in range(1, 17)),
    ]
    fake_launches = [
        ptb.Launch(
            cell_id=cell_id,
            command=(
                "fake-submit",
                cell_id,
                "--job-name",
                f"gangda_trial_0828.ptb.test.{cell_id}.formal.r1",
                "--hold",
            ),
            environment={
                "POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD": f"/evidence/{cell_id}.json",
                "POST_TRAIN_BENCH_CONTEXT_VALIDATION_SHA256": f"{index:064x}",
            },
        )
        for index, cell_id in enumerate(cell_ids, start=1)
    ]
    monkeypatch.setattr(ptb, "local_issues", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(ptb, "site_issues", list)
    monkeypatch.setattr(
        ptb,
        "source_snapshot",
        lambda: {
            "top_branch": "gangda_trial_0828",
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

    def fake_run(
        command: list[str] | tuple[str, ...], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
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
    assert len(submitted) == 32
    assert all("--hold" in command for command in submitted)
    assert commands[-1] == (
        "scontrol",
        "release",
        ",".join(str(job_id) for job_id in range(9001, 9033)),
    )
    assert receipt["state"] == "submitted"
    assert receipt["ownership"] == data["ownership"]
    assert receipt["source"]["top_branch"] == "gangda_trial_0828"
    assert len(receipt["jobs"]) == 32
    assert receipt["jobs"][0]["job_name"] == ("gangda_trial_0828.ptb.test.g01.formal.r1")
    assert set(receipt["context_validation"]) == set(cell_ids)
