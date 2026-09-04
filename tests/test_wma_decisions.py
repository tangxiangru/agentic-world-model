"""Public decision records preserve alternatives and the version actually chosen."""

from __future__ import annotations

import copy
import json
from itertools import pairwise

import pytest
from exp_protocol_cards import plan_card

from awm import wma_decisions as decisions
from awm.exp_protocol import schema as cards


def proposal_value():
    return {
        "schema_version": decisions.PROPOSAL_SCHEMA,
        "decision_id": "decision-01",
        "situation": {"remaining_h": 4, "incumbent": "models/best", "evidence": ["prior evaluation summary"]},
        "scientist_preference": "A",
        "candidates": [
            {
                "candidate_id": cid,
                "hypothesis": hypothesis,
                "parent_checkpoint": "models/best",
                "change": hypothesis,
                "train_h": cost,
                "eval_h": 0.2,
                "cost_basis": "measured throughput",
                "evidence": ["prior evaluation summary"],
                "uncertainty": "unobserved final accuracy",
                "decision_test": "same allowed evaluator as the incumbent",
            }
            for cid, hypothesis, cost in (
                ("A", "more epochs might help", 1.0),
                ("B", "new data might help", 1.4),
                ("C", "a different objective might help", 2.0),
            )
        ],
    }


def comparison_value(proposal):
    ranking = [candidate["candidate_id"] for candidate in reversed(proposal["candidates"])]
    return {
        "schema_version": decisions.COMPARISON_SCHEMA,
        "decision_id": proposal["decision_id"],
        "proposal_sha256": decisions.proposal_sha(proposal),
        "ranking": ranking,
        "comparisons": [
            {"preferred": a, "alternative": b, "reason": "within budget with useful new evidence",
             "evidence": ["shared prior-run summary"], "uncertainty": "complete-run quality remains unknown"}
            for a, b in pairwise(ranking)
        ],
        "candidate_assessments": [
            {"candidate_id": cid, "feasibility": "needs_check", "expected_effect": "uncertain",
             "opportunity_cost": "uses budget for an alternative", "uncertainty": "no full run yet"}
            for cid in ranking
        ],
        "suggestions": [{"candidate_id": "A", "kind": "optional_probe", "action": "verify the data mixture",
                         "evidence_scope": "row composition only, not final accuracy",
                         "decision_if_observed": "repair missing rows before launching the selected recipe"}],
    }


def session_card(tmp_path, *, state="delivered"):
    session = tmp_path / "session"
    card = plan_card()
    path = session / "memory" / "cards" / "exp-01.yaml"
    cards.dump_card(path, card)
    lock = {"plan_sha256": cards.plan_hash(card), "wma": {"state": state, "request_id": "request-01"}}
    path.with_suffix(".lock.json").write_text(json.dumps(lock))
    return session, path, lock


def test_valid_comparison_covers_three_real_candidates_and_both_adjacent_choices():
    proposal = proposal_value()
    comparison = comparison_value(proposal)
    decisions.validate_proposal(proposal)
    decisions.validate_comparison(comparison, proposal)
    assert proposal["scientist_preference"] == "A" and comparison["ranking"][0] == "C"


@pytest.mark.parametrize("field", ["train_h", "eval_h"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_proposed_costs_are_finite_nonnegative_numbers(field, value):
    proposal = proposal_value()
    proposal["candidates"][0][field] = value
    with pytest.raises(ValueError, match="finite nonnegative"):
        decisions.validate_proposal(proposal)


def test_singleton_is_permitted_only_with_an_explanation():
    proposal = proposal_value()
    proposal["candidates"] = proposal["candidates"][:1]
    with pytest.raises(ValueError, match="singleton_reason"):
        decisions.validate_proposal(proposal)
    proposal["singleton_reason"] = "Other real ideas require unavailable data."
    decisions.validate_proposal(proposal)
    value = comparison_value(proposal)
    assert value["comparisons"] == []
    decisions.validate_comparison(value, proposal)


@pytest.mark.parametrize("ranking", [["C", "B"], ["C", "B", "B"], ["C", "B", "unknown"], ["C", "B", {}]])
def test_ranking_cannot_omit_duplicate_or_invent_candidates(ranking):
    proposal = proposal_value()
    comparison = comparison_value(proposal)
    comparison["ranking"] = ranking
    with pytest.raises(ValueError, match="ranking"):
        decisions.validate_comparison(comparison, proposal)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reverse"])
def test_ranking_must_explain_every_actual_adjacent_pair(mutation):
    proposal = proposal_value()
    comparison = comparison_value(proposal)
    if mutation == "missing":
        comparison["comparisons"].pop()
    elif mutation == "duplicate":
        comparison["comparisons"][1] = copy.deepcopy(comparison["comparisons"][0])
    else:
        comparison["comparisons"].reverse()
    with pytest.raises(ValueError, match="adjacent"):
        decisions.validate_comparison(comparison, proposal)


@pytest.mark.parametrize("location", ["candidate_assessments", "suggestions"])
@pytest.mark.parametrize("invalid_id", ["unknown", {"id": "A"}])
def test_assessments_and_suggestions_reject_unknown_or_malformed_ids(location, invalid_id):
    proposal = proposal_value()
    comparison = comparison_value(proposal)
    comparison[location][0]["candidate_id"] = invalid_id
    with pytest.raises(ValueError):
        decisions.validate_comparison(comparison, proposal)


def test_a_changed_candidate_or_preference_invalidates_the_old_comparison():
    proposal = proposal_value()
    comparison = comparison_value(proposal)
    for mutate in (
        lambda p: p["candidates"][0].update(train_h=1.1),
        lambda p: p.update(scientist_preference="B"),
        lambda p: p["situation"].update(remaining_h=3.0),
    ):
        changed = copy.deepcopy(proposal)
        mutate(changed)
        with pytest.raises(ValueError, match="frozen proposal"):
            decisions.validate_comparison(comparison, changed)


def test_proposal_cannot_contain_the_candidate_own_observed_result():
    proposal = proposal_value()
    proposal["candidates"][0]["result"] = {"accuracy": 0.8}
    with pytest.raises(ValueError, match="unobserved proposals"):
        decisions.validate_proposal(proposal)


def test_hash_is_stable_across_object_key_order_but_tracks_candidate_order():
    proposal = proposal_value()
    reordered = {key: proposal[key] for key in reversed(list(proposal))}
    assert decisions.proposal_sha(reordered) == decisions.proposal_sha(proposal)
    changed = copy.deepcopy(proposal)
    changed["candidates"].reverse()
    assert decisions.proposal_sha(changed) != decisions.proposal_sha(proposal)


def test_new_proposals_never_replace_an_existing_editable_draft(tmp_path):
    session = tmp_path / "session"
    first = decisions.create_proposal(session)
    first.write_text(json.dumps(proposal_value()))
    before = first.read_bytes()
    second = decisions.create_proposal(session)
    assert first.name == "decision-01.proposal.json"
    assert second.name == "decision-02.proposal.json"
    assert first.read_bytes() == before
    assert decisions.load_proposal(session, "decision-01") == proposal_value()


def test_load_proposal_rejects_filename_identity_mismatch(tmp_path):
    session = tmp_path / "session"
    value = proposal_value()
    value["decision_id"] = "decision-02"
    decisions.write_once(decisions.proposal_path(session, "decision-01"), value)
    with pytest.raises(ValueError, match="filename"):
        decisions.load_proposal(session, "decision-01")


def test_write_once_keeps_first_record_and_rejects_non_json_before_creating_file(tmp_path):
    path = tmp_path / "archive" / "request.json"
    decisions.write_once(path, {"request": "original"})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        decisions.write_once(path, {"request": "replacement"})
    assert path.read_bytes() == before
    bad = tmp_path / "archive" / "bad.json"
    with pytest.raises(ValueError):
        decisions.write_once(bad, {"cost": float("nan")})
    assert not bad.exists()


def test_record_paths_cannot_escape_through_parent_or_symlink(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (session / "memory").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="leaves the session"):
        decisions.proposal_path(session, "decision-01")
    with pytest.raises(ValueError, match="leaves the session"):
        decisions.safe_path(session, "../outside/record.json")


@pytest.mark.parametrize("state", [None, "pending", "running"])
def test_proceed_action_waits_for_the_lock_review_to_return(tmp_path, state):
    session, _, _ = session_card(tmp_path, state=state)
    with pytest.raises(ValueError, match="not returned"):
        decisions.append_action(session, "exp-01", "proceed", "ready to start")
    assert decisions.latest_action(session, "exp-01") is None


def test_actions_require_a_lock_and_preserve_each_actual_choice(tmp_path):
    session, path, lock = session_card(tmp_path)
    lock_path = path.with_suffix(".lock.json")
    lock_path.unlink()
    with pytest.raises(ValueError, match="lock and await review"):
        decisions.append_action(session, "exp-01", "repair", "fix missing weights")
    lock_path.write_text(json.dumps(lock))
    first = decisions.append_action(session, "exp-01", "repair", "fix missing weights",
                                    suggestion="preconditions[0]", evidence=["save/load log"])
    first_bytes = first.read_bytes()
    second = decisions.append_action(session, "exp-01", "proceed", "save/load passed")
    assert first != second and first.read_bytes() == first_bytes
    assert decisions.latest_action(session, "exp-01")["action"] == "proceed"
    assert json.loads(first_bytes)["request_id"] == "request-01"


def test_action_fingerprints_recompute_live_script_and_preserve_old_version(tmp_path):
    session, path, lock = session_card(tmp_path)
    script = session / "train.py"
    script.write_text("old recipe\n")
    lock["script"] = {"path": "train.py", "sha256": "stale-stored-digest"}
    path.with_suffix(".lock.json").write_text(json.dumps(lock))
    first = decisions.append_action(session, "exp-01", "repair", "update the script")
    before = json.loads(first.read_text())
    assert before["fingerprint"]["files"][0]["sha256"] == cards.sha256_file(script)
    script.write_text("updated recipe\n")
    second = decisions.append_action(session, "exp-01", "repair", "script now updated")
    after = json.loads(second.read_text())
    assert before["fingerprint"]["files"] != after["fingerprint"]["files"]
    assert json.loads(first.read_text()) == before
