import json
from pathlib import Path

from awm.wma_evolve_hook import (
    analysis_prompt,
    claude_command,
    clean_cell_keys,
    discover_manifests,
    initial_state,
    should_trigger,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_discover_manifests_uses_newest_in_scope_receipt(tmp_path):
    manifest = tmp_path / "candidate.yaml"
    manifest.write_text("batch_id: test\n")
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    wrong = tmp_path / "wrong.json"
    base = {
        "batch_id": "wma-gsm8k-gemma4b-high-r04-a",
        "manifest": str(manifest),
        "subqueue": "gangda_wma_evolve",
    }
    _write(old, {**base, "submitted_at": "2026-09-03T01:00:00+00:00"})
    _write(new, {**base, "submitted_at": "2026-09-03T02:00:00+00:00"})
    _write(wrong, {**base, "subqueue": "gangda_exp-protocol-evolve"})
    registry = tmp_path / "registry.json"
    _write(
        registry,
        {
            "sources": [
                {"id": f"receipt:{old}", "kind": "receipt", "label": f"{base['batch_id']} [formal]", "registered_at": "1"},
                {"id": f"receipt:{new}", "kind": "receipt", "label": f"{base['batch_id']} [formal]", "registered_at": "2"},
                {"id": f"receipt:{wrong}", "kind": "receipt", "label": f"{base['batch_id']} [formal]", "registered_at": "3"},
            ]
        },
    )

    records = discover_manifests(registry)

    assert len(records) == 1
    assert records[0].receipt == new
    assert records[0].manifest == manifest


def test_clean_cells_exclude_flagged_or_incomplete_attempts():
    result = {
        "batch_id": "batch",
        "rows": [
            {"cell_id": "a", "complete": True, "completed_attempt": {"issues": [], "judge_flags": []}},
            {"cell_id": "b", "complete": True, "completed_attempt": {"issues": ["bad"], "judge_flags": []}},
            {"cell_id": "c", "complete": True, "completed_attempt": {"issues": [], "judge_flags": ["judge"]}},
            {"cell_id": "d", "complete": False, "completed_attempt": None},
        ],
    }

    assert clean_cell_keys(result) == {"batch/a"}


def test_trigger_is_eight_immediate_or_four_after_quiet_window():
    eight = {f"b/c{i}" for i in range(8)}
    assert should_trigger(
        eight, {cell: 100.0 for cell in eight}, now=101.0, min_new=8, min_partial=4, max_wait_seconds=10
    ) == (True, "new_clean_complete>=8")
    four = {f"b/c{i}" for i in range(4)}
    assert should_trigger(
        four, {cell: 100.0 for cell in four}, now=111.0, min_new=8, min_partial=4, max_wait_seconds=10
    ) == (True, "partial_clean_complete>=4_aged_10s")


def test_initial_state_acknowledges_preexisting_history():
    state = initial_state({"old/a", "old/b"})
    assert state["analyzed_cells"] == ["old/a", "old/b"]
    assert state["claims"] == {}


def test_claude_analysis_is_opus5_max_ultracode_and_read_only(tmp_path):
    command = claude_command(max_budget_usd=25)
    prompt = analysis_prompt(tmp_path / "payload.json", tmp_path / "snapshot.json")

    assert command[command.index("--model") + 1] == "claude-opus-5"
    assert command[command.index("--effort") + 1] == "max"
    assert command[command.index("--permission-mode") + 1] == "plan"
    assert prompt.startswith("ultracode\n")
    assert "Do not edit files" in prompt
    assert "Do not wait for or rely on GitHub PR comments" in prompt
