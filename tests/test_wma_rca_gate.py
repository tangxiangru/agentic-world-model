"""The lock-gate reader used for the online rounds' cost readout.

`tools/wma-rca/gate.py` is what turns a harvested cell into "how long the verdict-in-lock gate held
this run and how each lock ended". The Round 02 readout is preregistered on those numbers, so the
parser gets a transcript in the real shape: turn headers, a Bash call, its tool result.
"""

from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "wma-rca" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRANSCRIPT = """Session start — abc
  Working dir: /home/ben/task

Assistant — turn 2 | 2026-09-03T08:00:00Z
  Tool call — Bash (t1)
    {{
      "command": "awm exp_protocol lock --dir /home/ben/task exp-02 2>&1 | tail -30"
    }}

User — turn 2 | 2026-09-03T08:06:30Z
  Tool result — Bash (t1)
    locked exp-02
    WMA review requested for exp-02 (request r1); waiting up to 20 min — prepare the launch meanwhile, do not start it
    waiting for the WMA verdict on exp-02: 5.0 min elapsed
    waiting for the WMA verdict on exp-02: 5.5 min elapsed
    verdict: {verdict}
    verdict file: /home/ben/task/memory/cards/exp-02.verdict.json — read it before launching

Assistant — turn 3 | 2026-09-03T08:07:00Z
  Tool call — Bash (t2)
    {{
      "command": "nohup python train_sft.py --data data/sft_v1.jsonl --out ckpt/v1 > logs/t.log 2>&1 & echo started"
    }}

User — turn 3 | 2026-09-03T08:07:02Z
  Tool result — Bash (t2)
    started

Assistant — turn 4 | 2026-09-03T09:00:00Z
  Tool call — Bash (t3)
    {{
      "command": "awm exp_protocol lock --dir /home/ben/task exp-02 --relock \\"answering the precondition\\""
    }}

User — turn 4 | 2026-09-03T09:05:00Z
  Tool result — Bash (t3)
    note: this card was re-locked 1 time before the run
    WMA review requested for exp-02 (request r2); waiting up to 20 min
    waiting for the WMA verdict on exp-02: 4.5 min elapsed
    verdict: L0_runs=yes@0.9; L1_valid=yes@0.85; L2_effect=higher [0.3, 0.6]@0.6; L3_worth_now=yes@0.9
""".format(verdict="L0_runs=no@0.68; L1_valid=yes@0.72; L2_effect=higher [0.4, 0.72]@0.55; L3_worth_now=yes@0.7")

CONTROL = """Session start — def
  Working dir: /home/ben/task

Assistant — turn 2 | 2026-09-03T08:10:00Z
  Tool call — Bash (c1)
    {
      "command": "awm exp_protocol lock --dir /home/ben/task exp-01"
    }

User — turn 2 | 2026-09-03T08:10:04Z
  Tool result — Bash (c1)
    locked exp-01
    no world-model agent is attached to this session; no verdict
"""


@pytest.fixture
def batch(tmp_path: Path) -> Path:
    root = tmp_path / "results" / "ptb" / "wma-batch"
    for cell, text in (("w10r01", TRANSCRIPT), ("c10r01", CONTROL)):
        (root / cell).mkdir(parents=True)
        with gzip.open(root / cell / "solve_parsed.txt.gz", "wt") as handle:
            handle.write(text)
        (root / cell / "metrics.json").write_text('{"accuracy": 0.75}')
    return root


def test_each_lock_is_priced_and_its_outcome_read(batch: Path) -> None:
    gate = _load("gate")
    rows = gate.cell_rows("wma-batch", "w10r01", batch / "w10r01")

    assert [r["state"] for r in rows] == ["delivered", "delivered"]
    assert [r["gate_s"] for r in rows] == [390, 300]          # the wall time of each lock call
    assert [r["relock"] for r in rows] == [False, True]       # the second lock answers the precondition
    assert rows[0]["heartbeats"] == 2 and rows[0]["verdict"].startswith("L0_runs=no@0.68")
    assert rows[0]["launch_after_s"] == 30                    # training started after the verdict, not before

    agg = gate.aggregate(rows)
    assert agg["locks"] == 2 and agg["relocks"] == 1
    assert agg["gate_h_total"] == 0.19 and agg["gate_s_median"] == 345
    assert agg["by_cell"]["w10r01"]["cards"] == 1             # one card, two gates: the relock pays again


def test_a_control_cell_records_a_lock_with_no_agent_and_no_cost(batch: Path) -> None:
    gate = _load("gate")
    rows = gate.cell_rows("wma-batch", "c10r01", batch / "c10r01")

    assert [(r["state"], r["gate_s"]) for r in rows] == [("not_attached", 4)]
    assert gate.aggregate(rows)["gate_h_total"] == 0.0
