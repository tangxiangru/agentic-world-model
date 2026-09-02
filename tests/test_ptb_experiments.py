import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from awm import paths
from awm import ptb_experiments as ptb

MANIFEST = paths.REPO_ROOT / "experiments/posttrainbench/gsm8k-opus5-4x4-batch1.yaml"
DUAL_MANIFEST = (
    paths.REPO_ROOT / "experiments/posttrainbench/gsm8k-aime2025-opus5-4x4x2-batch1.yaml"
)
ROUND00_MANIFEST = (
    paths.REPO_ROOT
    / "experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-baseline-x16.yaml"
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
        Path(launch.environment["HF_HOME"]).resolve()
        == (paths.REPO_ROOT / "data/ptb/hf").resolve()
        for launch in launches
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
    assert data["batch_id"].endswith("batch1-v3")
    assert data["contract"]["run_index"] == 3
    assert [launch.cell_id for launch in launches] == [
        *(f"g{index:02d}" for index in range(1, 17)),
        *(f"a{index:02d}" for index in range(1, 17)),
    ]
    tasks = [launch.command[launch.command.index("--eval") + 1] for launch in launches]
    assert tasks == ["gsm8k"] * 16 + ["aime2025"] * 16
    assert all("--hold" in launch.command for launch in ptb.build_launches(data, hold=True))


def _selected_replication_manifest() -> dict:
    data = deepcopy(ptb.load_manifest(DUAL_MANIFEST))
    selected = data["cells"][:8] + data["cells"][16:24]
    cells = []
    for cell in selected:
        for replicate in (1, 2):
            copy = cell | {"id": f"{cell['id']}r{replicate}", "replicate": replicate}
            cells.append(copy)
    data["cells"] = cells
    data["contract"]["replication"] = {
        "settings": 16,
        "repeats": 2,
        "settings_per_task": 8,
    }
    data["pilot"]["cells"] = ["g06r1", "a06r1"]
    return data


def test_selected_replication_requires_sixteen_settings_with_two_repeats() -> None:
    data = _selected_replication_manifest()

    ptb.validate_manifest(data)
    launches = ptb.build_launches(data, hold=True)

    assert len(launches) == 32
    assert len({launch.cell_id for launch in launches}) == 32
    assert all("--hold" in launch.command for launch in launches)


def test_selected_replication_rejects_an_unbalanced_repeat() -> None:
    data = _selected_replication_manifest()
    data["cells"][-1] = data["cells"][0] | {"id": "extra", "replicate": 2}

    with pytest.raises(ptb.ExperimentError, match="repeated exactly 2 times"):
        ptb.validate_manifest(data)


def test_dual_task_pilot_covers_both_evaluation_paths() -> None:
    launches = ptb.build_launches(ptb.load_manifest(DUAL_MANIFEST), pilot=True)

    assert [launch.cell_id for launch in launches] == ["g06", "a06"]
    assert [launch.command[launch.command.index("--eval") + 1] for launch in launches] == [
        "gsm8k",
        "aime2025",
    ]
    assert all(launch.command[launch.command.index("--hours") + 1] == "1" for launch in launches)
    assert all("--hold" not in launch.command for launch in launches)


def test_explicit_retry_builds_only_selected_held_cells() -> None:
    launches = ptb.build_launches(
        ptb.load_manifest(DUAL_MANIFEST),
        cell_ids=["g01", "a01"],
        hold=True,
        purpose="formal-retry1",
    )

    assert [launch.cell_id for launch in launches] == ["g01", "a01"]
    assert all("--hold" in launch.command for launch in launches)
    assert all(
        any("formal-retry1" in argument for argument in launch.command) for launch in launches
    )


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
    with pytest.raises(ptb.ExperimentError, match="approved agent setup"):
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
    assert "missing or empty: judgement_general.json or its _rerun variant" in issues


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
            checkout=(
                {"sha": "3" * 40, "paths": ["awm"], "dir": "/vol/x", "digest": "4" * 64}
                if cell_id == "g01"
                else None
            ),
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
    ownership_registry = tmp_path / "slurm-ownership.json"
    monkeypatch.setattr(
        ptb,
        "read_ptb_env",
        lambda: {"POST_TRAIN_BENCH_SLURM_OWNERSHIP_REGISTRY": str(ownership_registry)},
    )
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
    assert receipt["awm_checkouts"] == {
        "g01": {"sha": "3" * 40, "paths": ["awm"], "dir": "/vol/x", "digest": "4" * 64}
    }
    registered = json.loads(ownership_registry.read_text(encoding="utf-8"))
    assert len(registered["sources"][0]["jobs"]) == 32


def test_pilot_receipt_is_registered_in_its_subqueue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _two_repeats_manifest()
    data["ownership"]["branch"] = "gangda_exp_protocol_evolve"
    fake_launch = ptb.Launch(
        cell_id="p01r1",
        command=(
            "fake-submit",
            "p01r1",
            "--job-name",
            "gangda_exp_protocol_evolve.ptb.ep-r00.p01r1.pilot.r1",
        ),
        environment={
            "POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD": "/evidence/p01r1.json",
            "POST_TRAIN_BENCH_CONTEXT_VALIDATION_SHA256": "1" * 64,
        },
    )
    monkeypatch.setattr(ptb, "local_issues", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(ptb, "site_issues", list)
    monkeypatch.setattr(
        ptb,
        "source_snapshot",
        lambda: {
            "top_branch": "gangda_exp_protocol_evolve",
            "top_commit": "1" * 40,
            "ptb_commit": "2" * 40,
            "top_status": "",
            "ptb_status": "",
        },
    )
    monkeypatch.setattr(ptb, "dry_run", lambda *_args, **_kwargs: [])
    ownership_registry = tmp_path / "slurm-ownership.json"
    monkeypatch.setattr(
        ptb,
        "read_ptb_env",
        lambda: {
            "POST_TRAIN_BENCH_SLURM_OWNERSHIP_REGISTRY": str(ownership_registry),
            "POST_TRAIN_BENCH_SLURM_SUBQUEUE": "gangda_exp-protocol-evolve",
        },
    )
    monkeypatch.setattr(ptb.paths, "data_root", lambda: tmp_path)
    monkeypatch.setattr(ptb, "build_launches", lambda *_args, **_kwargs: [fake_launch])
    monkeypatch.setattr(
        ptb.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, "Submitted Slurm job 9001\n", ""
        ),
    )

    receipt_path = ptb.submit(data, pilot=True)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    registry = json.loads(ownership_registry.read_text(encoding="utf-8"))
    assert receipt["subqueue"] == "gangda_exp-protocol-evolve"
    assert registry["sources"][0]["subqueue"] == "gangda_exp-protocol-evolve"
    assert registry["sources"][0]["jobs"][0]["job_id"] == "9001"


def test_root_owned_allocations_are_released_through_sudo() -> None:
    assert ptb._release_command({"POST_TRAIN_BENCH_SLURM_SUBMIT_AS_ROOT": "1"}, "10,11") == [
        "sudo",
        "scontrol",
        "release",
        "10,11",
    ]
    assert ptb._release_command({}, "10,11") == ["scontrol", "release", "10,11"]


def test_site_gate_accepts_the_exp_protocol_two_node_subqueue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from awm import slurm_queue

    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(slurm_queue._default_registry()), encoding="utf-8")
    nodes = ["slurm2-a3nodesetondem-0", "slurm2-a3nodesetondem-1"]
    monkeypatch.setattr(
        ptb,
        "read_ptb_env",
        lambda: {
            "POST_TRAIN_BENCH_SLURM_GPU_MODE": "gres",
            "POST_TRAIN_BENCH_SLURM_PARTITION": "ptb-a3",
            "POST_TRAIN_BENCH_SLURM_NODELIST": "slurm2-a3nodesetondem-[0-1]",
            "POST_TRAIN_BENCH_SLURM_SUBMIT_AS_ROOT": "0",
            "POST_TRAIN_BENCH_SLURM_SUBQUEUE": "gangda_exp-protocol-evolve",
            "POST_TRAIN_BENCH_SLURM_OWNERSHIP_REGISTRY": str(registry),
        },
    )
    monkeypatch.setattr(ptb, "current_top_branch", lambda: "gangda_exp_protocol_evolve")

    def fake_check_output(command: list[str], **_kwargs: object) -> str:
        if command[:3] == ["scontrol", "show", "partition"]:
            return "PartitionName=ptb-a3 OverSubscribe=NO\n"
        if command[:3] == ["scontrol", "show", "hostnames"]:
            return "\n".join(nodes) + "\n"
        if command[:3] == ["scontrol", "show", "node"]:
            return f"NodeName={command[3]} CfgTRES=cpu=104,gres/gpu=8\n"
        raise AssertionError(command)

    monkeypatch.setattr(ptb.subprocess, "check_output", fake_check_output)

    assert ptb.site_issues() == []


def test_site_gate_rejects_nodes_outside_the_named_subqueue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from awm import slurm_queue

    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(slurm_queue._default_registry()), encoding="utf-8")
    monkeypatch.setattr(
        ptb,
        "read_ptb_env",
        lambda: {
            "POST_TRAIN_BENCH_SLURM_GPU_MODE": "gres",
            "POST_TRAIN_BENCH_SLURM_PARTITION": "ptb-a3",
            "POST_TRAIN_BENCH_SLURM_NODELIST": "slurm2-a3nodesetondem-[0,2]",
            "POST_TRAIN_BENCH_SLURM_SUBMIT_AS_ROOT": "0",
            "POST_TRAIN_BENCH_SLURM_SUBQUEUE": "gangda_exp-protocol-evolve",
            "POST_TRAIN_BENCH_SLURM_OWNERSHIP_REGISTRY": str(registry),
        },
    )
    monkeypatch.setattr(ptb, "current_top_branch", lambda: "gangda_exp_protocol_evolve")

    def fake_check_output(command: list[str], **_kwargs: object) -> str:
        if command[:3] == ["scontrol", "show", "partition"]:
            return "PartitionName=ptb-a3 OverSubscribe=NO\n"
        if command[:3] == ["scontrol", "show", "hostnames"]:
            return "slurm2-a3nodesetondem-0\nslurm2-a3nodesetondem-2\n"
        if command[:3] == ["scontrol", "show", "node"]:
            return f"NodeName={command[3]} CfgTRES=cpu=104,gres/gpu=8\n"
        raise AssertionError(command)

    monkeypatch.setattr(ptb.subprocess, "check_output", fake_check_output)

    assert ptb.site_issues() == [
        (
            "site nodelist for subqueue gangda_exp-protocol-evolve must be "
            "slurm2-a3nodesetondem-0,slurm2-a3nodesetondem-1; got "
            "slurm2-a3nodesetondem-0,slurm2-a3nodesetondem-2"
        )
    ]


# ---- a batch is any set of approved cells (2026-09-01) ------------------------
# The first batches were a full 4x4 matrix and a 16x2 replication; those were
# decisions about those batches, not properties of a valid batch. A protocol-
# iteration round is two repeats of one setting, and must submit through the
# same launcher, receipts, and registry.


def _two_repeats_manifest() -> dict:
    data = deepcopy(ptb.load_manifest(MANIFEST))
    gemma_max = next(cell for cell in data["cells"] if cell["id"] == "b04")
    data["batch_id"] = "ep-r01-baseline-x2"
    data["cells"] = [gemma_max | {"id": f"p01r{r}", "replicate": r} for r in (1, 2)]
    data["contract"]["replication"] = {"settings": 1, "repeats": 2}
    data["contract"]["base_models"] = {
        "google/gemma-3-4b-pt": data["contract"]["base_models"]["google/gemma-3-4b-pt"]
    }
    data["pilot"] = {"cell": "p01r1", "agent_budget_hours": 1}
    return data


def test_two_repeats_of_one_setting_is_a_valid_batch() -> None:
    data = _two_repeats_manifest()
    ptb.validate_manifest(data)
    launches = ptb.build_launches(data, hold=True)
    assert [launch.cell_id for launch in launches] == ["p01r1", "p01r2"]
    assert all(
        launch.environment["POST_TRAIN_BENCH_BASE_MODEL_REVISION"]
        == "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
        for launch in launches
    )
    assert [launch.command[launch.command.index("--job-name") + 1] for launch in launches] == [
        "gangda_trial_0828.ptb.ep-r01-baseline-x2.p01r1.formal.r1",
        "gangda_trial_0828.ptb.ep-r01-baseline-x2.p01r2.formal.r1",
    ]
    (pilot,) = ptb.build_launches(data, pilot=True)
    assert pilot.cell_id == "p01r1"
    assert pilot.command[pilot.command.index("--hours") + 1] == "1"


def test_a_batch_pins_only_the_base_models_it_uses_but_all_of_those() -> None:
    data = _two_repeats_manifest()
    ptb.validate_manifest(data)
    data["cells"][0]["base_model"] = "Qwen/Qwen3-4B-Base"
    with pytest.raises(ptb.ExperimentError, match="does not pin"):
        ptb.validate_manifest(data)
    data["contract"]["base_models"]["not/approved"] = {"revision": "a" * 40}
    with pytest.raises(ptb.ExperimentError, match="approved starting models"):
        ptb.validate_manifest(data)


def test_a_cell_outside_the_approved_setups_is_rejected() -> None:
    data = _two_repeats_manifest()
    data["cells"][0]["agent"] = "claude_non_api"
    with pytest.raises(ptb.ExperimentError, match="approved agent setup"):
        ptb.validate_manifest(data)


def test_high_effort_awm_scaffold_is_an_approved_setup() -> None:
    data = _awm_manifest()
    for cell in data["cells"]:
        cell["agent"] = "claude_vertex_high_awm"
        cell["effort"] = "high"

    ptb.validate_manifest(data)

    launches = ptb.build_launches(data)
    assert all(
        launch.command[launch.command.index("--agent") + 1]
        == "claude_vertex_high_awm"
        for launch in launches
    )


def test_round00_is_sixteen_identical_high_awm_baseline_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ptb.paths, "data_root", lambda *_args, **_kwargs: tmp_path)

    data = ptb.load_manifest(ROUND00_MANIFEST)
    launches = ptb.build_launches(data, hold=True)

    assert data["ownership"] == {
        "branch": "gangda_exp_protocol_evolve",
        "spec": "doc/spec/2026-09-02-exp-protocol-round00-gsm8k-baseline.md",
    }
    assert data["contract"]["task"] == "gsm8k"
    assert data["contract"]["replication"] == {"settings": 1, "repeats": 16}
    assert data["contract"]["official_judge_container"] == "opus_5.sif"
    assert len(launches) == 16
    assert [cell["replicate"] for cell in data["cells"]] == list(range(1, 17))
    assert {cell["agent"] for cell in data["cells"]} == {"claude_vertex_high_awm"}
    assert {cell["effort"] for cell in data["cells"]} == {"high"}
    assert {cell["base_model"] for cell in data["cells"]} == {"google/gemma-3-4b-pt"}
    assert {cell["awm"]["sha"] for cell in data["cells"]} == {
        "eaf50919ff5f79f15e33df7bb49f44ffebacfc64"
    }
    assert all(cell["awm"]["paths"] == list(ptb.EXP_PROTOCOL_SHIP) for cell in data["cells"])
    assert all("--hold" in launch.command for launch in launches)
    assert all(
        launch.environment["POST_TRAIN_BENCH_OFFICIAL_JUDGE_CONTAINER_SHA256"]
        == "35f287e7b17d62ab44cd95db26dfeeac166943daed5f7b557b008bae51acc759"
        for launch in launches
    )
    (pilot,) = ptb.build_launches(data, pilot=True)
    assert pilot.cell_id == "p00r01"
    assert pilot.command[pilot.command.index("--hours") + 1] == "1"


def test_duplicate_cell_ids_are_rejected() -> None:
    data = _two_repeats_manifest()
    data["cells"][1]["id"] = "p01r1"
    with pytest.raises(ptb.ExperimentError, match="unique"):
        ptb.validate_manifest(data)


def test_an_empty_batch_is_rejected() -> None:
    data = _two_repeats_manifest()
    data["cells"] = []
    del data["pilot"]
    with pytest.raises(ptb.ExperimentError, match="at least one cell"):
        ptb.validate_manifest(data)


def test_replication_is_checked_against_the_shape_it_declares() -> None:
    data = _two_repeats_manifest()
    data["contract"]["replication"] = {"settings": 1, "repeats": 3}
    with pytest.raises(ptb.ExperimentError, match="repeated exactly 3 times"):
        ptb.validate_manifest(data)
    data["contract"]["replication"] = {"settings": 1, "repeats": 2}
    data["cells"][1]["replicate"] = 1
    with pytest.raises(ptb.ExperimentError, match="replicates 1..2"):
        ptb.validate_manifest(data)


def test_pilot_is_optional_but_a_pilot_launch_needs_one() -> None:
    data = _two_repeats_manifest()
    del data["pilot"]
    ptb.validate_manifest(data)
    with pytest.raises(ptb.ExperimentError, match="no pilot"):
        ptb.build_launches(data, pilot=True)
    data["pilot"] = {"cell": "nope", "agent_budget_hours": 1}
    with pytest.raises(ptb.ExperimentError, match="existing"):
        ptb.validate_manifest(data)


def test_a_single_task_batch_may_be_aime2025() -> None:
    data = _two_repeats_manifest()
    data["contract"]["task"] = "aime2025"
    ptb.validate_manifest(data)
    launch, _ = ptb.build_launches(data)
    assert launch.command[launch.command.index("--eval") + 1] == "aime2025"


def test_a_cell_task_outside_the_batch_tasks_is_rejected() -> None:
    data = _two_repeats_manifest()
    data["cells"][0]["task"] = "aime2025"
    with pytest.raises(ptb.ExperimentError, match="not one of the batch"):
        ptb.validate_manifest(data)


def test_a_task_outside_the_approved_list_is_rejected() -> None:
    data = _two_repeats_manifest()
    data["contract"]["task"] = "humaneval"
    with pytest.raises(ptb.ExperimentError, match="subset"):
        ptb.validate_manifest(data)


def test_run_index_is_any_positive_integer() -> None:
    data = _two_repeats_manifest()
    data["contract"]["run_index"] = 7
    ptb.validate_manifest(data)
    launch, _ = ptb.build_launches(data)
    assert launch.command[launch.command.index("--job-name") + 1].endswith(".r7")
    data["contract"]["run_index"] = 0
    with pytest.raises(ptb.ExperimentError, match="positive integer"):
        ptb.validate_manifest(data)


# ---- cells that ship a checkout of this repository (2026-09-02) ---------------
# An `_awm` scaffold mounts a read-only checkout at /home/ben/awm and runs
# `awm sandbox setup` before the prompt. The cell says which commit, which
# paths, and which setup arguments; the launcher materialises the archive on
# the data volume and hands the scaffold the bind and the two variables.


def _awm_manifest() -> dict:
    data = _two_repeats_manifest()
    sha = ptb._git(ptb.paths.REPO_ROOT, "rev-parse", "HEAD")
    for cell in data["cells"]:
        cell["agent"] = "claude_vertex_max_awm"
        cell["awm"] = {
            "sha": sha,
            "paths": list(ptb.EXP_PROTOCOL_SHIP),
            "setup": "--exp-protocol --tool claude",
            "protocol_tree": ptb._git(
                ptb.paths.REPO_ROOT, "rev-parse", "HEAD:skills/exp_protocol"
            ),
        }
    return data


def test_an_awm_cell_ships_its_checkout_read_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ptb.paths, "data_root", lambda *_a, **_k: tmp_path)
    data = _awm_manifest()
    ptb.validate_manifest(data)
    launches = ptb.build_launches(data, hold=True)
    sha = data["cells"][0]["awm"]["sha"]
    first = launches[0]
    checkout = Path(first.environment["POST_TRAIN_BENCH_EXTRA_BINDS"].split(":")[0])
    assert first.environment["POST_TRAIN_BENCH_EXTRA_BINDS"] == f"{checkout}:/home/ben/awm:ro"
    assert first.environment["AWM_SANDBOX_SETUP"] == "--exp-protocol --tool claude"
    assert first.environment["AWM_CHECKOUT_SHA"] == sha
    assert checkout.is_relative_to(tmp_path / "ptb" / "awm-checkouts")
    assert (checkout / "awm" / "cli.py").is_file()
    assert (checkout / "skills" / "exp_protocol" / "SKILL.md").is_file()
    assert not (checkout / "skills" / "exp_protocol_meta").exists()
    assert not (checkout / "doc").exists()
    assert not (checkout / "awm" / "ptb_ops.py").exists()
    assert not (checkout / "awm" / "traj").exists()
    assert first.checkout == {
        "sha": sha,
        "paths": list(ptb.EXP_PROTOCOL_SHIP),
        "dir": str(checkout),
        "digest": first.checkout["digest"],
        "protocol_tree": ptb._git(ptb.paths.REPO_ROOT, "rev-parse", "HEAD:skills/exp_protocol"),
        "setup": "--exp-protocol --tool claude",
    }
    assert len(first.checkout["digest"]) == 64
    # the second cell, same sha and paths, reuses the same materialised directory
    assert launches[1].checkout["dir"] == str(checkout)
    marker = json.loads((checkout / ".awm-checkout.json").read_text())
    assert marker["sha"] == sha and marker["digest"] == first.checkout["digest"]


def test_a_plain_cell_gets_no_checkout_variables(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ptb.paths, "data_root", lambda *_a, **_k: tmp_path)
    launches = ptb.build_launches(_two_repeats_manifest())
    assert "POST_TRAIN_BENCH_EXTRA_BINDS" not in launches[0].environment
    assert launches[0].checkout is None


def test_an_awm_cell_needs_its_block_and_a_plain_cell_must_not_have_one() -> None:
    data = _awm_manifest()
    del data["cells"][0]["awm"]
    with pytest.raises(ptb.ExperimentError, match="must declare an awm block"):
        ptb.validate_manifest(data)
    data = _awm_manifest()
    data["cells"][0]["agent"] = "claude_vertex_max"
    with pytest.raises(ptb.ExperimentError, match="would ignore it"):
        ptb.validate_manifest(data)


@pytest.mark.parametrize(
    "paths",
    [["skills"], ["doc"], ["."], ["../awm"], ["/awm"], ["skills/exp_protocol_meta"], [], ["awm", ""],
     ["awm"], ["awm/wma"], ["skills/wma"], ["skills/wma_meta"], ["awm/wma/estimator.py"]],
)
def test_awm_paths_may_not_reach_the_meta_skill_the_wma_or_the_docs(paths: list[str]) -> None:
    data = _awm_manifest()
    data["cells"][0]["awm"]["paths"] = paths
    with pytest.raises(ptb.ExperimentError, match="awm.paths"):
        ptb.validate_manifest(data)


def test_awm_sha_and_setup_are_checked() -> None:
    data = _awm_manifest()
    data["cells"][0]["awm"]["sha"] = "1db6a9e"
    with pytest.raises(ptb.ExperimentError, match="full commit"):
        ptb.validate_manifest(data)
    data = _awm_manifest()
    data["cells"][0]["awm"]["setup"] = ""
    with pytest.raises(ptb.ExperimentError, match="awm.setup"):
        ptb.validate_manifest(data)


def test_awm_issues_name_a_missing_commit_or_path() -> None:
    sha = ptb._git(ptb.paths.REPO_ROOT, "rev-parse", "HEAD")
    good = {
        "sha": sha,
        "paths": list(ptb.EXP_PROTOCOL_SHIP),
        "setup": "--exp-protocol",
        "protocol_tree": ptb.protocol_tree_at(sha),
    }
    assert ptb._awm_issues("p01r1", good) == []
    missing_commit = good | {"sha": "f" * 40}
    assert any("not in this repository" in issue for issue in ptb._awm_issues("p01r1", missing_commit))
    missing_path = good | {"paths": ["awm/cli.py", "skills/no_such_skill"]}
    assert any("skills/no_such_skill" in issue for issue in ptb._awm_issues("p01r1", missing_path))


def test_materialising_a_checkout_is_idempotent_and_refuses_the_meta_skill(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(ptb.paths, "data_root", lambda *_a, **_k: tmp_path)
    sha = ptb._git(ptb.paths.REPO_ROOT, "rev-parse", "HEAD")
    first = ptb.materialize_awm_checkout(sha, list(ptb.EXP_PROTOCOL_SHIP))
    stamp = (Path(first["dir"]) / ".awm-checkout.json").stat().st_mtime_ns
    second = ptb.materialize_awm_checkout(sha, list(ptb.EXP_PROTOCOL_SHIP))
    assert second == first
    assert (Path(first["dir"]) / ".awm-checkout.json").stat().st_mtime_ns == stamp
    with pytest.raises(ptb.ExperimentError, match="exp_protocol_meta"):
        ptb.materialize_awm_checkout(sha, ["skills"])
    with pytest.raises(ptb.ExperimentError, match="awm/wma"):
        ptb.materialize_awm_checkout(sha, ["awm"])
    with pytest.raises(ptb.ExperimentError, match="not in this repository"):
        ptb.materialize_awm_checkout("f" * 40, ["awm/cli.py"])


# ---- the protocol tree (2026-09-02) ---------------------------------------------
# The iteration line names a variant by the tree of skills/exp_protocol, not by
# a commit: the commit a cell ships must carry the setup step, which the commit
# that last touched the skill may predate, and a later commit is the same
# variant only while that tree is unchanged.


def test_awm_protocol_tree_names_the_variant_the_cell_ships(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ptb.paths, "data_root", lambda *_a, **_k: tmp_path)
    sha = ptb._git(ptb.paths.REPO_ROOT, "rev-parse", "HEAD")
    tree = ptb._git(ptb.paths.REPO_ROOT, "rev-parse", "HEAD:skills/exp_protocol")
    assert ptb.protocol_tree_at(sha) == tree
    assert ptb.protocol_tree_at("f" * 40) is None
    good = {
        "sha": sha,
        "paths": list(ptb.EXP_PROTOCOL_SHIP),
        "setup": "--exp-protocol",
        "protocol_tree": tree,
    }
    assert ptb._awm_issues("p01r1", good) == []
    other = good | {"protocol_tree": "e" * 40}
    assert any("protocol_tree" in issue for issue in ptb._awm_issues("p01r1", other))
    data = _awm_manifest()
    del data["cells"][0]["awm"]["protocol_tree"]
    assert any(
        "protocol_tree must declare" in issue
        for issue in ptb._awm_issues("p01r1", data["cells"][0]["awm"])
    )
    data = _awm_manifest()
    data["cells"][0]["awm"]["protocol_tree"] = tree[:7]
    with pytest.raises(ptb.ExperimentError, match="protocol_tree"):
        ptb.validate_manifest(data)
    data["cells"][0]["awm"]["protocol_tree"] = tree
    ptb.validate_manifest(data)
    # the materialised checkout, and so the receipt, carry the tree they ship
    checkout = ptb.materialize_awm_checkout(sha, list(ptb.EXP_PROTOCOL_SHIP))
    assert checkout["protocol_tree"] == tree
    marker = json.loads((Path(checkout["dir"]) / ".awm-checkout.json").read_text())
    assert marker["protocol_tree"] == tree
    assert ptb.materialize_awm_checkout(sha, list(ptb.EXP_PROTOCOL_SHIP))["protocol_tree"] == tree
    assert ptb.materialize_awm_checkout(sha, ["awm/cli.py"])["protocol_tree"] is None


def test_a_marker_behind_the_launcher_is_upgraded_in_place_never_rebuilt(
    tmp_path: Path, monkeypatch
) -> None:
    # Pilot 90462 (2026-09-02) lost `awm` mid-run to a stale NFS handle: the launcher
    # replaced its bind-mounted checkout to add protocol_tree to the marker. The bytes are
    # fixed by (sha, paths); only the marker moves.
    monkeypatch.setattr(ptb.paths, "data_root", lambda *_a, **_k: tmp_path)
    sha = ptb._git(ptb.paths.REPO_ROOT, "rev-parse", "HEAD")
    shipped = list(ptb.EXP_PROTOCOL_SHIP)
    first = ptb.materialize_awm_checkout(sha, shipped)
    checkout = Path(first["dir"])
    marker = checkout / ".awm-checkout.json"
    stale = json.loads(marker.read_text())
    stale.pop("protocol_tree")
    marker.write_text(json.dumps(stale) + "\n")
    sentinel = checkout / "awm" / "cli.py"
    inode_before = checkout.stat().st_ino
    mtime_before = sentinel.stat().st_mtime_ns

    upgraded = ptb.materialize_awm_checkout(sha, shipped)

    expected = ptb.protocol_tree_at(sha)
    assert upgraded["protocol_tree"] == expected
    assert upgraded["dir"] == str(checkout) and upgraded["digest"] == first["digest"]
    assert checkout.stat().st_ino == inode_before  # the directory itself survived
    assert sentinel.stat().st_mtime_ns == mtime_before  # and so did its files
    info = json.loads(marker.read_text())
    assert info["protocol_tree"] == expected and info["marker_upgraded_at"]


def test_a_complete_checkout_is_never_deleted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ptb.paths, "data_root", lambda *_a, **_k: tmp_path)
    sha = ptb._git(ptb.paths.REPO_ROOT, "rev-parse", "HEAD")
    shipped = list(ptb.EXP_PROTOCOL_SHIP)
    checkout = Path(ptb.materialize_awm_checkout(sha, shipped)["dir"])
    marker = checkout / ".awm-checkout.json"
    info = json.loads(marker.read_text())
    info["sha"] = "0" * 40  # corrupt: a complete marker that disagrees with the directory name
    marker.write_text(json.dumps(info) + "\n")
    with pytest.raises(ptb.ExperimentError, match="refusing to replace"):
        ptb.materialize_awm_checkout(sha, shipped)
    assert (checkout / "awm" / "cli.py").is_file()
    info["sha"] = sha
    info["protocol_tree"] = "0" * 40
    marker.write_text(json.dumps(info) + "\n")
    with pytest.raises(ptb.ExperimentError, match="corrupt marker"):
        ptb.materialize_awm_checkout(sha, shipped)
    assert (checkout / "awm" / "cli.py").is_file()
    marker.write_text("{not json\n")
    with pytest.raises(ptb.ExperimentError, match="unreadable marker"):
        ptb.materialize_awm_checkout(sha, shipped)
    assert (checkout / "awm" / "cli.py").is_file()
    # a half-written directory (no complete marker) is still replaced
    marker.unlink()
    rebuilt = ptb.materialize_awm_checkout(sha, shipped)
    assert Path(rebuilt["dir"]) == checkout and (checkout / ".awm-checkout.json").is_file()
