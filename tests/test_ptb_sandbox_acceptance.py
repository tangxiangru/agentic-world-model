"""CPU-only acceptance of the sandbox wiring: what a scientist cell would actually get.

Materialise the checkout the launcher would bind, run the scaffold's setup
step against it the way solve.sh does (PYTHONPATH = the checkout, cwd = the
task directory, no AWM_EXP_PROTOCOL_DIR), then drive one card through the
protocol with the checkout's own code. Along the way: the meta skill and the
docs are nowhere the scientist can reach.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from awm import paths
from awm import ptb_experiments as ptb
from awm.exp_protocol import lineage
from test_exp_protocol_cli import fill_plan, fill_result, write_jsonl

SHIPPED = ["awm", "skills/exp_protocol"]


def _sandbox_env(checkout: Path) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("AWM_EXP_PROTOCOL_DIR", "AWM_DATA_ROOT", "PYTHONPATH")
    }
    env["PYTHONPATH"] = str(checkout)
    return env


def _awm(checkout: Path, task: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "awm.cli", *args],
        cwd=task,
        env=_sandbox_env(checkout),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def cell(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ptb.paths, "data_root", lambda *_a, **_k: tmp_path / "vol")
    sha = ptb._git(paths.REPO_ROOT, "rev-parse", "HEAD")
    checkout = Path(ptb.materialize_awm_checkout(sha, SHIPPED)["dir"])
    task = tmp_path / "home" / "ben" / "task"
    task.mkdir(parents=True)
    return sha, checkout, task


def test_the_scientist_sees_the_protocol_and_nothing_about_its_iteration(cell) -> None:
    sha, checkout, task = cell
    assert not (checkout / "skills" / "exp_protocol_meta").exists()
    assert not (checkout / "doc").exists()
    assert not list(checkout.rglob("*exp_protocol_meta*"))

    done = _awm(checkout, task, "sandbox", "setup", "--target", str(task), "--sha", sha,
                "--exp-protocol", "--tool", "claude")
    assert done.returncode == 0, done.stdout + done.stderr

    where = subprocess.run(
        [sys.executable, "-c", "import awm; print(awm.__file__)"],
        cwd=task, env=_sandbox_env(checkout), capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert Path(where).is_relative_to(checkout), "the sandbox must import the shipped checkout"

    assert (task / ".claude" / "skills" / "exp_protocol" / "SKILL.md").is_file()
    assert "skills/exp_protocol/SKILL.md" in (task / "CLAUDE.md").read_text()
    assert json.loads((task / "awm_sandbox.json").read_text())["sha"] == sha
    assert not list(task.rglob("*exp_protocol_meta*"))
    assert not (task / "AGENTS.md").exists()  # --tool claude only


def test_one_card_end_to_end_with_the_shipped_code(cell) -> None:
    sha, checkout, task = cell
    assert _awm(checkout, task, "sandbox", "setup", "--target", str(task), "--sha", sha,
                "--exp-protocol", "--tool", "claude").returncode == 0
    (task / "train.py").write_text("pass\n")
    write_jsonl(task / "data" / "train.jsonl",
                [{"prompt": "q", "completion": "a #### 1<|im_end|>"} for _ in range(20)])
    (task / "ckpts").mkdir()
    d = str(task)

    assert _awm(checkout, task, "exp_protocol", "new", "--dir", d).returncode == 0
    card_path = lineage.cards_dir(task) / "exp-01.yaml"
    assert _awm(checkout, task, "exp_protocol", "check", "--dir", d, "exp-01").returncode == 1
    fill_plan(card_path, task)
    assert _awm(checkout, task, "exp_protocol", "check", "--dir", d, "exp-01").returncode == 0
    locked = _awm(checkout, task, "exp_protocol", "lock", "--dir", d, "exp-01")
    assert locked.returncode == 0, locked.stdout + locked.stderr
    assert (lineage.cards_dir(task) / "exp-01.lock.json").is_file()
    assert (lineage.cards_dir(task) / "exp-01.preflight.json").is_file()
    fill_result(card_path, task)
    closed = _awm(checkout, task, "exp_protocol", "close", "--dir", d, "exp-01")
    assert closed.returncode == 0, closed.stdout + closed.stderr
    assert "| exp-01 |" in (task / "memory" / "index.md").read_text()

    # the layout the harvest mirrors: <cell>/task/memory/cards, metrics.json beside task/
    (task.parent / "metrics.json").write_text('{"accuracy": 0.5, "stderr": 0.01}')
    collected = _awm(checkout, task, "exp_protocol", "collect", d, "--csv")
    assert collected.returncode == 0
    header, row = collected.stdout.strip().splitlines()
    assert header.startswith("session,accuracy,n_cards")
    assert row.startswith("ben/task,0.5,1,")
