"""reconcile: the verdict meets the outcome, in place or from a truth file kept elsewhere."""

from __future__ import annotations

import pytest

from awm.exp_protocol import schema as cards
from awm.wma import backends, reconcile, review, schema
from exp_protocol_cards import closed_card, plan_card


def reviewed_session(tmp_path, skill_dir):
    s = tmp_path / "session"
    cards.dump_card(s / "memory" / "cards" / "exp-01.yaml", plan_card())
    review.review(s, "exp-01", backends.HeuristicBackend(), mode="offline", skill_dir=skill_dir)
    return s


@pytest.fixture
def skill(tmp_path):
    d = tmp_path / "skill"
    d.mkdir()
    (d / "SKILL.md").write_text("x")
    return d


def test_reconcile_in_place_after_close(tmp_path, skill) -> None:
    s = reviewed_session(tmp_path, skill)
    cards.dump_card(s / "memory" / "cards" / "exp-01.yaml", closed_card())
    v = reconcile.reconcile(s / "memory" / "cards" / "exp-01.yaml")
    assert v["actual"]["decision"] == "adopt" and v["scored"]["L0"] == "hit" and v["reconciled_at"]
    assert schema.load_verdict(schema.verdict_path(s / "memory" / "cards" / "exp-01.yaml"))["scored"] == v["scored"]


def test_reconcile_from_a_truth_card_outside_the_session(tmp_path, skill) -> None:
    s = reviewed_session(tmp_path, skill)
    truth = tmp_path / "_truth" / "exp-01.yaml"
    done = closed_card()
    done["result"]["execution"] = "failed"
    done["result"]["output_checkpoint"] = None
    done["result"]["measurements"] = []
    done["conclusion"].update({"verdict": "inconclusive", "decision": "reject"})
    cards.dump_card(truth, done)
    v = reconcile.reconcile(s / "memory" / "cards" / "exp-01.yaml", truth_path=truth)
    assert v["scored"]["L0"] == "miss" and v["scored"]["L3"] == "miss"
    # the session's own card is still open: nothing leaked back into it
    assert "result" not in cards.load_card(s / "memory" / "cards" / "exp-01.yaml")


def test_reconcile_without_a_verdict_raises(tmp_path) -> None:
    p = tmp_path / "memory" / "cards" / "exp-01.yaml"
    cards.dump_card(p, closed_card())
    with pytest.raises(reconcile.ReconcileError):
        reconcile.reconcile(p)


def test_reconcile_without_truth_is_unscorable_not_an_error(tmp_path, skill) -> None:
    s = reviewed_session(tmp_path, skill)
    v = reconcile.reconcile(s / "memory" / "cards" / "exp-01.yaml")
    assert set(v["scored"].values()) == {"unscorable"}
