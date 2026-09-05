"""The actual launch boundary binds an explicit decision to a reviewed plan."""

from __future__ import annotations

import json
import sys

import pytest
from exp_protocol_cards import plan_card

from awm import paths, wma_client
from awm.cli import main
from awm.exp_protocol import decisions, lock, schema
from awm.wma import backends, sidecar


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("AWM_EXP_PROTOCOL_DIR", str(paths.REPO_ROOT / "skills/exp_protocol"))
    root = tmp_path / "session"
    cards = root / "memory/cards"
    cards.mkdir(parents=True)
    (root / "config.json").write_text('{"value": 1}\n')
    (root / "execute.py").write_text("from pathlib import Path\nPath('launched.txt').write_text('ran')\n")
    card = plan_card()
    card["setup"]["method"] = {"family": "other"}
    card["setup"]["data"] = [{"path": str(root / "config.json"), "source": "test configuration", "n_examples": 1}]
    card["setup"]["command"] = {"argv": [sys.executable, str(root / "execute.py")],
        "cwd": str(root), "script": str(root / "execute.py"), "configs": [str(root / "config.json")]}
    card["setup"]["output_dir"] = str(root / "output")
    card["setup"]["parent_checkpoint"]["path"] = "google/gemma-3-4b-pt"
    schema.dump_card(cards / "exp-01.yaml", card)
    return root


def lock_and_proceed(root):
    assert main(["exp_protocol", "lock", "--dir", str(root), "exp-01"]) == 0
    assert main(["wma", "act", "--dir", str(root), "exp-01", "--action", "proceed", "--reason", "checked current plan"]) == 0


def test_single_mode_training_needs_review_and_action_but_no_fake_choice(session):
    from awm import sandbox
    sandbox.setup(session, sha="test", exp_protocol=True, decision_mode="single")
    path = session / "memory/cards/exp-01.yaml"
    card = schema.load_card(path)
    data = session / "data.jsonl"
    data.write_text('{"completion":"answer #### 1<|im_end|>"}\n')
    card["setup"]["data"] = [{"path": str(data), "source": "synthetic", "n_examples": 1}]
    card["setup"]["method"] = {"family": "sft", "stop_token": "<|im_end|>",
        "answer_marker": "#### ", "hyperparams": {"max_seq_len": 2048}}
    schema.dump_card(path, card)
    assert main(["exp_protocol", "run", "--dir", str(session), "exp-01"]) == 2
    lock_and_proceed(session)
    assert main(["exp_protocol", "run", "--dir", str(session), "exp-01"]) == 0
    assert not (session / "memory/decisions").exists()


def test_mode_change_invalidates_no_wma_control_proceed(session):
    from awm import sandbox
    from awm.exp_protocol import treatment
    sandbox.setup(session, sha="test", exp_protocol=True, decision_mode="multi-self")
    lock_and_proceed(session)
    path = session / "awm_sandbox.json"
    value = json.loads(path.read_text())
    value.update(decision_mode="single",
                 decision_mode_sha256=treatment.describe("single", explicit=True)["sha256"])
    path.write_text(json.dumps(value))
    assert main(["exp_protocol", "run", "--dir", str(session), "exp-01"]) == 2
    assert not (session / "launched.txt").exists()


def test_run_executes_only_after_lock_and_current_proceed_and_cannot_reuse_action(session):
    root = session
    command = ["exp_protocol", "run", "--dir", str(root), "exp-01"]
    assert main(command) == 2
    assert not (root / "launched.txt").exists()
    assert main(["exp_protocol", "lock", "--dir", str(root), "exp-01"]) == 0
    assert main(command) == 2
    assert main(["wma", "act", "--dir", str(root), "exp-01", "--action", "proceed", "--reason", "ready"]) == 0
    assert main(command) == 0
    assert (root / "launched.txt").read_text() == "ran"
    assert main(command) == 2
    events = list((root / ".wma/launches/exp-01").glob("*.exit.json"))
    assert len(events) == 1
    assert json.loads(events[0].read_text())["scientific_completion"] == "not_assessed"


@pytest.mark.parametrize("mutation", ["script", "config", "plan"])
def test_plan_or_pinned_file_change_cannot_launch_under_old_decision(session, mutation):
    lock_and_proceed(session)
    if mutation == "script":
        (session / "execute.py").write_text("raise RuntimeError('modified')\n")
    elif mutation == "config":
        (session / "config.json").write_text('{"value": 2}\n')
    else:
        path = session / "memory/cards/exp-01.yaml"
        card = schema.load_card(path)
        card["hypothesis"]["claim"] = "different plan"
        schema.dump_card(path, card)
    assert main(["exp_protocol", "run", "--dir", str(session), "exp-01"]) == 2
    assert not (session / "launched.txt").exists()


def test_even_unchanged_relock_requires_a_new_decision(session):
    lock_and_proceed(session)
    assert main(["exp_protocol", "lock", "--dir", str(session), "exp-01", "--relock", "confirm current version"]) == 0
    assert main(["exp_protocol", "run", "--dir", str(session), "exp-01"]) == 2
    assert not (session / "launched.txt").exists()


def test_pending_review_cannot_be_bypassed_by_proceed(session):
    path = session / "memory/cards/exp-01.yaml"
    lock.write_lock(path, schema.load_card(path), {})
    assert main(["wma", "act", "--dir", str(session), "exp-01", "--action", "proceed", "--reason", "premature"]) == 2
    assert main(["exp_protocol", "run", "--dir", str(session), "exp-01"]) == 2


def test_same_card_verdict_from_another_request_does_not_authorize_launch(session):
    lock_and_proceed(session)
    path = session / "memory/cards/exp-01.yaml"
    fingerprint = decisions.card_fingerprint(session, "exp-01")
    lock.annotate_lock(path, "wma", {"state": "delivered", "request_id": "expected", "fingerprint": fingerprint})
    path.with_suffix(".verdict.json").write_text(json.dumps({"request_id": "other", "review_fingerprint": fingerprint}))
    decisions.append_action(session, "exp-01", "proceed", "this request")
    assert main(["exp_protocol", "run", "--dir", str(session), "exp-01"]) == 2
    assert not (session / "launched.txt").exists()


def test_training_requires_comparison_choice_even_without_sidecar(session):
    path = session / "memory/cards/exp-01.yaml"
    card = schema.load_card(path)
    data = session / "data.jsonl"
    data.write_text('{"completion":"answer #### 1<|im_end|>"}\n')
    card["setup"]["data"] = [{"path": str(data), "source": "synthetic", "n_examples": 1}]
    card["setup"]["method"] = {"family": "sft", "stop_token": "<|im_end|>",
        "answer_marker": "#### ", "hyperparams": {"max_seq_len": 2048}}
    schema.dump_card(path, card)
    lock_and_proceed(session)
    command = ["exp_protocol", "run", "--dir", str(session), "exp-01"]
    assert main(command) == 2
    proposal = json.loads((paths.REPO_ROOT / "skills/exp_protocol/proposal.example.json").read_text())
    did = proposal["decision_id"]
    decisions.write_once(decisions.proposal_path(session, did), proposal)
    assert wma_client.compare_and_wait(session, did)["state"] == "not_attached"
    wma_client.record_choice(session, did, proposal["scientist_preference"], "self-selected control", "exp-01")
    assert main(command) == 0


def test_sidecar_archives_distinct_requests_and_rejects_plan_changed_in_queue(session, tmp_path):
    (session / ".wma/requests").mkdir(parents=True)
    private = tmp_path / "private"
    skill = private / "skills/wma"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("test WMA")
    path = session / "memory/cards/exp-01.yaml"
    card = schema.load_card(path)
    lock.write_lock(path, card, {})
    first, _ = wma_client.enqueue(session, ["exp-01"])
    config = sidecar.Config(session, skill, None, "heuristic", "test", "high", backends.Budget(), 1, private)
    sidecar.run(config, once=True)
    first_copy = session / f".wma/reviews/{first}/exp-01/verdict.json"
    before = first_copy.read_bytes()
    second, _ = wma_client.enqueue(session, ["exp-01"])
    sidecar.run(config, once=True)
    assert first_copy.read_bytes() == before
    assert json.loads((session / "memory/cards/exp-01.verdict.json").read_text())["request_id"] == second
    third, _ = wma_client.enqueue(session, ["exp-01"])
    card["hypothesis"]["claim"] = "edited while queued"
    schema.dump_card(path, card)
    sidecar.run(config, once=True)
    response = json.loads((session / f".wma/responses/{third}.json").read_text())
    assert response["state"] == "failed" and "changed after" in response["errors"]["exp-01"]
    assert not list((session / "memory/cards").glob("*.transcript*.jsonl"))
