"""The Stop hook is run, not grepped: JSON in, decision out."""

from __future__ import annotations

import json
import subprocess
import sys

from awm.exp_protocol import lock, preflight, schema
from exp_protocol_cards import closed_card, plan_card

HOOK = preflight.skill_dir() / "hooks" / "stop_open_cards.py"


def run_hook(session, payload: dict) -> dict:
    out = subprocess.run([sys.executable, str(HOOK)], input=json.dumps({"cwd": str(session), **payload}),
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def locked(session, card):
    path = session / "memory" / "cards" / f"{card['card_id']}.yaml"
    schema.dump_card(path, card)
    lock.write_lock(path, card, {})
    return path


def test_a_locked_card_without_a_conclusion_blocks_until_closed_and_says_why(tmp_path) -> None:
    """Cell p00r08 ended its turn with training at step 207/4154; the session died with it."""
    path = locked(tmp_path, plan_card())
    first = run_hook(tmp_path, {})
    assert first["decision"] == "block"
    assert "ENDS THE SESSION" in first["reason"] and "exp-01" in first["reason"]
    assert "block 1 of" in first["reason"]
    # a second attempt to stop is blocked again: stop_hook_active does not end the guard
    again = run_hook(tmp_path, {"stop_hook_active": True})
    assert again["decision"] == "block" and "block 2 of" in again["reason"]
    # closing the card by hand (result + conclusion in the YAML) releases it
    card = closed_card()
    schema.dump_card(path, card)
    assert run_hook(tmp_path, {"stop_hook_active": True}) == {}


def test_the_guard_is_bounded_so_a_broken_tool_cannot_hold_the_session_forever(tmp_path) -> None:
    from exp_protocol_cards import plan_card as _plan
    locked(tmp_path, _plan())
    import importlib.util
    spec = importlib.util.spec_from_file_location("stop_open_cards", HOOK)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for n in range(1, mod.MAX_BLOCKS + 1):
        out = run_hook(tmp_path, {})
        assert out["decision"] == "block" and f"block {n} of {mod.MAX_BLOCKS}" in out["reason"]
    assert run_hook(tmp_path, {}) == {}
    assert json.loads((tmp_path / "memory" / ".stop_hook.json").read_text())["blocks"] == mod.MAX_BLOCKS


def test_a_closed_card_does_not_block(tmp_path) -> None:
    locked(tmp_path, closed_card())
    assert run_hook(tmp_path, {}) == {}


def test_a_copy_of_the_template_is_not_mistaken_for_a_closed_card(tmp_path) -> None:
    """I11: the template literally contains `decision: adopt | reject | ...`."""
    path = tmp_path / "memory" / "cards" / "exp-01.yaml"
    path.parent.mkdir(parents=True)
    path.write_text((preflight.skill_dir() / "card.template.yaml").read_text())
    lock.write_lock(path, schema.load_card(path), {})
    assert run_hook(tmp_path, {})["decision"] == "block"


def test_decision_null_or_in_prose_is_still_open(tmp_path) -> None:
    card = plan_card()
    card["conclusion"] = {"summary": "the decision: adopt is pending", "decision": None}
    locked(tmp_path, card)
    assert run_hook(tmp_path, {})["decision"] == "block"


def test_a_malformed_card_does_not_crash_the_hook(tmp_path) -> None:
    path = tmp_path / "memory" / "cards" / "exp-01.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("just a string\n")
    (tmp_path / "memory" / "cards" / "exp-01.lock.json").write_text("{}")
    assert run_hook(tmp_path, {})["decision"] == "block"


def test_no_session_dir_is_a_pass_through(tmp_path) -> None:
    assert run_hook(tmp_path / "nowhere", {}) == {}
