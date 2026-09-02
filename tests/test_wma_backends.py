"""The heuristic backend is the baseline and the test double; a command backend is any agent that writes the file."""

from __future__ import annotations

import json
import os
import stat

import pytest

from awm.exp_protocol import schema as cards
from awm.wma import backends, schema
from exp_protocol_cards import plan_card


def brief(tmp_path, card=None) -> backends.Brief:
    card = card or plan_card()
    card_path = tmp_path / "memory" / "cards" / f"{card['card_id']}.yaml"
    cards.dump_card(card_path, card)
    return backends.Brief(card_id=card["card_id"], session_dir=tmp_path, card_path=card_path,
                          verdict_path=schema.verdict_path(card_path), skill_dir=tmp_path / "skills" / "wma",
                          mode="offline", budget=backends.Budget(cpu_min=1, gpu_min=0, wall_min=1),
                          model=None, prompt="write the verdict")


def test_heuristic_writes_a_valid_verdict_for_a_training_card(tmp_path) -> None:
    b = brief(tmp_path)
    backends.HeuristicBackend().run(b)
    v = schema.load_verdict(b.verdict_path)
    assert schema.validate_verdict(v).ok, schema.validate_verdict(v).render()
    assert v["backend"] == "heuristic" and v["mode"] == "offline"
    assert v["levels"]["L2_effect"]["interval"][0] <= v["levels"]["L2_effect"]["interval"][1]
    assert v["levels"]["L0_runs"]["basis"] and v["evidence"]


def test_heuristic_defers_late_in_the_run_and_widens_for_non_training_families(tmp_path) -> None:
    card = plan_card()
    card["situation"]["elapsed_h"] = 9.0
    card["setup"]["method"] = {"family": "merge"}
    b = brief(tmp_path, card)
    backends.HeuristicBackend().run(b)
    v = schema.load_verdict(b.verdict_path)
    assert v["levels"]["L3_worth_now"]["answer"] == "defer"
    lo, hi = v["levels"]["L2_effect"]["interval"]
    assert hi - lo > 0.05


def fake_executable(tmp_path, body: str):
    exe = tmp_path / "fake-agent"
    exe.write_text("#!/bin/bash\n" + body)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def test_command_backend_accepts_an_agent_that_writes_the_file(tmp_path) -> None:
    b = brief(tmp_path)
    good = json.dumps({**schema.empty_verdict("exp-01"), "levels": {
        "L0_runs": {"answer": "yes", "confidence": 0.9, "basis": []},
        "L1_valid": {"answer": "yes", "confidence": 0.9, "basis": []},
        "L2_effect": {"metric": "accuracy", "direction": "higher", "interval": [0.0, 0.05], "confidence": 0.5, "basis": []},
        "L3_worth_now": {"answer": "yes", "confidence": 0.6, "expected_cost_h": 1.0, "basis": []}}})
    exe = fake_executable(tmp_path, f"cat > /dev/null\nmkdir -p $(dirname '{b.verdict_path}')\ncat > '{b.verdict_path}' <<'J'\n{good}\nJ\n")
    be = backends.CommandBackend("fake", [str(exe), "--model", "{model}"], model="m1")
    be.run(b)
    v = schema.load_verdict(b.verdict_path)
    assert v["backend"] == "fake" and v["levels"]["L0_runs"]["answer"] == "yes"


def test_command_backend_rejects_an_agent_that_writes_nothing(tmp_path) -> None:
    b = brief(tmp_path)
    exe = fake_executable(tmp_path, "cat > /dev/null\necho did nothing\n")
    with pytest.raises(backends.BackendError, match="no verdict"):
        backends.CommandBackend("fake", [str(exe)]).run(b)


def test_command_backend_rejects_an_invalid_verdict(tmp_path) -> None:
    b = brief(tmp_path)
    exe = fake_executable(tmp_path, f"cat > /dev/null\nmkdir -p $(dirname '{b.verdict_path}')\necho '{{\"schema_version\": \"x\"}}' > '{b.verdict_path}'\n")
    with pytest.raises(backends.BackendError, match="invalid"):
        backends.CommandBackend("fake", [str(exe)]).run(b)


def test_command_backend_times_out_on_the_wall_budget(tmp_path) -> None:
    b = brief(tmp_path)
    b.budget = backends.Budget(cpu_min=1, gpu_min=0, wall_min=0.01)
    exe = fake_executable(tmp_path, "sleep 5\n")
    with pytest.raises(backends.BackendError, match="timed out"):
        backends.CommandBackend("fake", [str(exe)]).run(b)


def test_registry_names_the_three_backends_and_passes_the_model(tmp_path) -> None:
    assert set(backends.BACKENDS) == {"heuristic", "claude", "codex"}
    be = backends.get_backend("claude", model="claude-opus-5")
    assert isinstance(be, backends.CommandBackend) and "claude-opus-5" in be.argv()
    with pytest.raises(backends.BackendError):
        backends.get_backend("nope")
    assert os.path.basename(be.argv()[0]) == "claude"


def test_an_empty_model_leaves_no_dangling_model_flag() -> None:
    for model in (None, ""):
        argv = backends.get_backend("claude", model=model).argv()
        assert "--model" not in argv and "{model}" not in " ".join(argv), argv
