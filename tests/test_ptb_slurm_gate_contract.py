"""Zero-GPU regression checks for PostTrainBench Slurm gate orchestration."""

from pathlib import Path


RUN_GATES = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "PostTrainBench"
    / "src"
    / "commit_utils"
    / "slurm"
    / "run_gates.sh"
)
GATE_WORKER = RUN_GATES.with_name("gate_worker.sbatch")
GPU_REAP = GATE_WORKER.parents[2] / "utils" / "gpu_reap.sh"


def gate_case(script: str, name: str, next_name: str) -> str:
    start = script.index(f"    {name})")
    end = script.index(f"    {next_name})", start)
    return script[start:end]


def test_g1_requires_an_observed_concurrent_running_window():
    script = RUN_GATES.read_text(encoding="utf-8")
    g1 = gate_case(script, "g1", "g2")

    concurrency_check = '[ "$running" -eq 2 ]'
    assert 'squeue -h -j "$jobs" -t R' in g1
    assert concurrency_check in g1
    assert g1.index(concurrency_check) < g1.index('wait_jobs "$jobs"')
    assert "both G1 canaries were never RUNNING concurrently" in g1


def test_g1_checks_the_complete_allocation_shape():
    script = RUN_GATES.read_text(encoding="utf-8")
    g1 = gate_case(script, "g1", "g2")

    assert "$3 != 16" in g1
    assert "/mem=(128G|131072M)(,|$)/" in g1
    assert "/gres\\/gpu=1(,|$)/" in g1


def test_gate_uuid_query_uses_cgroup_visible_logical_gpu():
    worker = GATE_WORKER.read_text(encoding="utf-8")

    assert "nvidia-smi --query-gpu=uuid" in worker
    assert 'nvidia-smi -i "$ALLOCATED"' not in worker
    assert '[ "${#VISIBLE_UUIDS[@]}" -eq 1 ]' in worker


def test_gpu_reap_uses_the_devices_cgroup_as_its_gres_scope():
    script = GPU_REAP.read_text(encoding="utf-8")

    assert 'if [ "${POST_TRAIN_BENCH_SLURM_GPU_MODE:-}" != "gres" ]; then' in script
    assert 'device_args=(-i "$gpu_selector")' in script
    assert 'nvidia-smi "${device_args[@]}" --query-compute-apps=pid' in script


def test_failed_gate_cancels_only_jobs_recorded_by_this_invocation():
    script = RUN_GATES.read_text(encoding="utf-8")

    assert "SUBMITTED_JOB_IDS=()" in script
    assert "trap cleanup_submitted_jobs EXIT" in script
    assert 'scancel "${SUBMITTED_JOB_IDS[@]}"' in script
    for variable in ("first", "second", "job", "ninth", "survivor", "reaper"):
        assert f'SUBMITTED_JOB_IDS+=("${variable}")' in script


def test_g3_reuses_its_gpu_after_reaping_without_harming_the_survivor():
    runner = RUN_GATES.read_text(encoding="utf-8")
    worker = GATE_WORKER.read_text(encoding="utf-8")
    g3 = gate_case(runner, "g3", "*")

    assert "reaper-survivor-proof.txt" in g3
    assert "reaper-eval-smoke.txt" in g3
    assert worker.index("ptb_reap_allocated_gpu_processes") < worker.index("reaper-eval-smoke.txt")
    assert "torch.cuda.synchronize()" in worker
