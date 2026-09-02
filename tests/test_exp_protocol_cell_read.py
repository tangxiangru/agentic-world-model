"""The cell reader parses a real harvested bundle: both Bash-call renderings, the
lock-before-launch timing, and the inspect-log sample estimate."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "exp_protocol_cell_read.py"
BUNDLE = REPO / "results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-x16-v3/p00r05"
RELOCKED_BUNDLE = REPO / "results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-x16-v3/p00r07"


def _load():
    spec = importlib.util.spec_from_file_location("cell_read", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_commands_reads_both_renderings():
    mod = _load()
    lines = [
        "Assistant — turn 2 | 2026-09-02T12:25:29Z",
        "  Tool call — Bash (toolu_1)",
        "    $ ls -la && bash timer.sh",
        "Assistant — turn 9 | 2026-09-02T13:00:00Z",
        "  Tool call — Bash (toolu_2)",
        "    {",
        '      "command": nohup python scripts/train_sft.py --out /home/ben/task/ckpts/exp-03 > logs/exp-03.log 2>&1 &',
        "    echo launched",
        '      "description": "Launch exp-03",',
        '      "timeout": 1200000',
        "    }",
    ]
    cmds = mod.commands(lines)
    assert len(cmds) == 2
    assert cmds[0][2] == "ls -la && bash timer.sh"
    assert cmds[0][1].isoformat().startswith("2026-09-02T12:25:29")
    assert cmds[1][2].startswith("nohup python scripts/train_sft.py")
    assert "echo launched" in cmds[1][2]
    assert cmds[1][1].isoformat().startswith("2026-09-02T13:00:00")


@pytest.mark.skipif(not BUNDLE.exists(), reason="the p00r05 bundle is not checked out")
def test_reads_a_real_bundle():
    out = subprocess.run(
        [sys.executable, str(TOOL), str(BUNDLE)], text=True, capture_output=True, check=True
    ).stdout
    assert "accuracy=0.7043214556482184" in out
    assert "-- cards: 10" in out
    assert "lock_before_launch=3/3" in out
    assert "overrides total=1 relocks=1" in out
    assert "max 504" in out  # dev-500 inspect logs at ~44 KB per sample
    assert "time_taken=06:17:15" in out


@pytest.mark.skipif(not RELOCKED_BUNDLE.exists(), reason="the p00r07 bundle is not checked out")
def test_a_close_time_relock_does_not_turn_the_original_launch_into_a_violation():
    out = subprocess.run(
        [sys.executable, str(TOOL), str(RELOCKED_BUNDLE)],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "exp-05:" in out
    assert "launch 17:12:36Z AFTER lock 17:12:32Z" in out
    assert "lock_before_launch=5/5" in out
