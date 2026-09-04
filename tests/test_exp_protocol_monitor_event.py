"""A wake-up reports terminal identities, not scientific validation or authority."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def event(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "tools/exp_protocol_monitor_event.py"
    spec = importlib.util.spec_from_file_location("monitor_event_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "tools/exp_protocol_monitor_event.py"))
    path = tmp_path / "data/ptb/monitor/exp_protocol_goal.json"
    path.parent.mkdir(parents=True)
    return module, path


@pytest.mark.parametrize("payload", [None, "not json", [], 42, {"status": "watching"}])
def test_no_ready_event_emits_nothing(event, capsys, payload):
    module, path = event
    if payload is not None:
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    assert module.main() == 0
    assert capsys.readouterr().out == ""


def test_ready_event_preserves_terminal_ids_and_validation_boundary(event, capsys):
    module, path = event
    state = {"status": "ready", "terminal_count": 6, "threshold": 6,
             "terminal_jobs": ["101", "102"], "checked_at": "2026-09-04T08:27:36Z"}
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert payload["hookEventName"] == "SessionStart"
    context = payload["additionalContext"]
    assert "6/6" in context and "jobs=101,102" in context
    assert "eight NEW clean cells" in context and "predeclared discovery/confirmation block" in context
    assert "Terminal count alone proves neither condition" in context
    assert "do not launch filler repeats" in context
    assert path.read_bytes() == before
