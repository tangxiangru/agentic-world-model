"""Contract tests for the deliberately thin PostTrainBench study wrapper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from rollout import build_prompts, study_matrix
from rollout.patches import apply_extra_binds, apply_prompt_file


REPO = Path(__file__).resolve().parent.parent
ROLLOUT = REPO / "rollout"


def test_matrix_is_exactly_24_one_gpu_cells_without_opus_46() -> None:
    cells = study_matrix.study_matrix()
    assert len(cells) == 24
    assert len({cell.spec for cell in cells}) == 24
    assert {cell.scientist_model for cell in cells} == {
        "claude-opus-4-8",
        "claude-opus-5",
    }
    assert all("claude-opus-4-6" not in cell.spec for cell in cells)
    assert all(cell.record()["num_hours"] == 10 for cell in cells)


def test_prompts_are_ptb_plus_only_the_declared_study_sections() -> None:
    base = "before\n\n## Rules\n1. baseline\n"
    c1 = build_prompts.ptb_fulltraj(base)
    c2 = build_prompts.wm_prompt(base, fulltraj=True)
    c3 = build_prompts.wm_prompt(base, fulltraj=False)

    assert "## Prior runs" in c1
    assert "## The world-model agent" not in c1
    assert "## Prior runs" in c2 and "## The world-model agent" in c2
    assert "## Prior runs" not in c3 and "## The world-model agent" in c3
    for prompt in (c1, c2, c3):
        assert "## Pinned base checkpoint" not in prompt
        assert prompt.count("## Rules") == 1


def test_c1_scientist_invocation_matches_ptb_claude_baseline() -> None:
    solve = (ROLLOUT / "agents/claude_fulltraj_noawm/solve.sh").read_text()
    for fragment in (
        "bash /home/ben/update_agent_cli.sh claude",
        'claude --print --verbose --model "$AGENT_CONFIG"',
        "--output-format stream-json --thinking-display summarized",
        "--dangerously-skip-permissions",
    ):
        assert fragment in solve


@pytest.mark.parametrize("agent", ("claude_fulltraj_noawm", "claude_wm"))
def test_agents_have_no_custom_rollout_gate(agent: str) -> None:
    solve = (ROLLOUT / f"agents/{agent}/solve.sh").read_text()
    for removed in (
        "sanitize_result_tree.py",
        "redact_claude_stream.py",
        "attest_claude_runtime.py",
        "validate_base_model_cache.py",
        "validate_c1_final_model.py",
        "validate_study_corpus.py",
        "validate_wma_session.py",
        "release_gate.py",
    ):
        assert removed not in solve
    assert "tee " not in solve


def test_wma_returns_scientist_status_and_peer_cleanup_is_best_effort() -> None:
    solve = (ROLLOUT / "agents/claude_wm/solve.sh").read_text()
    assert 'WMA_MODEL="${AWM_WMA_MODEL:-claude-opus-5}"' in solve
    assert "scientist_rc=$?" in solve
    assert 'exit "${scientist_rc}"' in solve
    assert 'kill "${WMA_PID}" 2>/dev/null || true' in solve


def test_prompt_file_patch_is_small_selective_and_idempotent() -> None:
    original = 'x\necho "$PROMPT" > "${EVAL_DIR}/prompt.txt"\ny\n'
    patched = apply_prompt_file.apply(original)
    assert "claude_fulltraj_noawm|claude_wm" in patched
    assert 'cp "${EVAL_DIR}/prompt.txt" "${JOB_DIR}/task/instruction.md"' in patched
    assert apply_prompt_file.apply(patched) == patched


def test_extra_bind_patch_remains_idempotent() -> None:
    original = (
        'x\n'
        '    timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" \\\n'
        '    apptainer exec \\\n'
        '        --bind "${HF_MERGED}:${HF_HOME_NEW}" \\\n'
        'y\n'
    )
    patched = apply_extra_binds.apply(original)
    assert "POST_TRAIN_BENCH_EXTRA_BINDS" in patched
    assert apply_extra_binds.apply(patched) == patched


def _fake_ptb(tmp_path: Path) -> tuple[Path, Path, Path]:
    ptb = tmp_path / "ptb"
    (ptb / "src/commit_utils").mkdir(parents=True)
    (ptb / "src/commit_utils/set_env_vars.sh").write_text(":\n")
    capture = tmp_path / "args.txt"
    env_capture = tmp_path / "env.txt"
    (ptb / "src/run_task.sh").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n'
        'env | sort > "$CAPTURE_ENV"\n'
    )
    return ptb, capture, env_capture


@pytest.mark.parametrize(
    ("spec", "input_name", "agent", "config", "prompt", "mount"),
    (
        (
            "c1:claude-opus-4-8:train:1",
            "PRIOR_RUNS",
            "claude_fulltraj_noawm",
            "claude-opus-4-8",
            "prompt_fulltraj",
            "/home/ben/prior_runs:ro",
        ),
        (
            "c2:claude-opus-5:traj:train,test:2",
            "PRIOR_RUNS",
            "claude_wm",
            "claude-opus-5:traj:train,test",
            "prompt_wm_fulltraj",
            "/home/ben/prior_runs:ro",
        ),
        (
            "c3:claude-opus-4-8:retrieval:train:1",
            "WM_MEMORY",
            "claude_wm",
            "claude-opus-4-8:retrieval:train",
            "prompt_wm",
            "/home/ben/wm-memory:ro",
        ),
    ),
)
def test_pack_calls_ptb_directly_with_one_gpu(
    tmp_path: Path,
    spec: str,
    input_name: str,
    agent: str,
    config: str,
    prompt: str,
    mount: str,
) -> None:
    ptb, capture, env_capture = _fake_ptb(tmp_path)
    study_input = tmp_path / "study-input"
    study_input.mkdir()
    env = {
        **os.environ,
        "HV_PTB_DIR": str(ptb),
        input_name: str(study_input),
        "CAPTURE_ARGS": str(capture),
        "CAPTURE_ENV": str(env_capture),
        "PTB_RUN_ID": "test",
        "PTB_GPU_SLOTS": "7",
    }
    subprocess.run(
        ["bash", str(ROLLOUT / "wm_pack.sbatch"), spec],
        cwd=REPO,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    args = capture.read_text().splitlines()
    assert args[0:3] == ["gsm8k", agent, "google/gemma-3-4b-pt"]
    assert args[4:] == ["10", config, "1"]
    forwarded = env_capture.read_text()
    assert f"POST_TRAIN_BENCH_PROMPT={prompt}\n" in forwarded
    assert mount in forwarded
    assert "NUM_GPUS=1\n" in forwarded
    assert "POST_TRAIN_BENCH_VISIBLE_GPUS=7\n" in forwarded


def test_setup_installs_only_mechanical_ptb_extensions() -> None:
    setup = (ROLLOUT / "setup.sh").read_text()
    assert "apply_extra_binds.py" in setup
    assert "apply_prompt_file.py" in setup
    assert 'archive --format=tar "${AWM_REPO_COMMIT}" awm input wma' in setup
    assert "--exclude=awm/credential_guard.py" in setup
    assert "--exclude=awm/wm/agents/llm.py" in setup
    for removed in (
        "apply_study_runner.py",
        "apply_scratch_root.py",
        "attest_study_surface.py",
        "validate_study_corpus.py",
        "validate_base_model_cache.py",
        "release_gate.py",
    ):
        assert removed not in setup


def test_removed_runtime_gate_files_stay_removed() -> None:
    removed = (
        "attest_claude_runtime.py",
        "attest_ptb_surface.py",
        "release_gate.py",
        "validate_base_model_cache.py",
        "validate_c1_final_model.py",
        "validate_study_corpus.py",
        "validate_wma_session.py",
        "patches/apply_scratch_root.py",
        "patches/apply_study_runner.py",
    )
    assert all(not (ROLLOUT / name).exists() for name in removed)
