"""The timeline tool attributes tool execution time to categories and finds the stages."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "exp_protocol_trace_timeline.py"
BUNDLE = REPO / "results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-x16-v3/p00r05"


def _load():
    spec = importlib.util.spec_from_file_location("trace_timeline", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_and_classify_synthetic_turns():
    mod = _load()
    lines = [
        "Assistant — turn 1 | 2026-09-02T12:00:00Z",
        "  Tool call — Bash (t1)",
        "    $ awm exp_protocol lock --dir /home/ben/task exp-02",
        "User — turn 1 | 2026-09-02T12:00:10Z",
        "  Tool result — Bash (t1)",
        "Assistant — turn 2 | 2026-09-02T12:01:00Z",
        "  Tool call — Bash (t2)",
        "    $ nohup python scripts/train_sft.py --out ckpts/exp-02 > logs/exp-02.log 2>&1 &",
        "User — turn 2 | 2026-09-02T12:01:05Z",
        "Assistant — turn 3 | 2026-09-02T12:02:00Z",
        "  Tool call — Bash (t3)",
        "    $ sleep 900; tail -n 3 logs/exp-02.log",
        "User — turn 3 | 2026-09-02T12:17:00Z",
        "Assistant — turn 4 | 2026-09-02T12:18:00Z",
        "  Tool call — Write (t4)",
        "    {",
        '      "file_path": "/home/ben/task/memory/cards/exp-03.yaml",',
        "    }",
        "User — turn 4 | 2026-09-02T12:18:02Z",
    ]
    ev = mod.parse(lines)
    assert [e[1] for e in ev if e[0] == "Assistant"] == [1, 2, 3, 4]
    assert mod.classify("Bash", ev[0][4]) == "protocol"
    assert mod.classify("Bash", ev[2][4]) == "train_launch"
    assert mod.classify("Bash", ev[4][4]) == "waiting_on_runs"
    assert mod.classify("Write", ev[6][4]) == "protocol"


@pytest.mark.skipif(not BUNDLE.exists(), reason="the p00r05 bundle is not checked out")
def test_reads_a_real_bundle():
    out = subprocess.run([sys.executable, str(TOOL), str(BUNDLE)], text=True, capture_output=True, check=True).stdout
    assert "p00r05: 12:27Z -> 18:43Z = 6.28 h" in out
    assert "waiting_on_runs" in out and "protocol" in out
    assert "first_lock" in out and "first_train_launch" in out
