"""Selection-stage comparisons use one frozen input and one attributable call."""

from __future__ import annotations

import copy
import json
from itertools import pairwise

import pytest

from awm import wma_decisions as decisions
from awm.wma import backends, compare, schema


@pytest.fixture
def proposal():
    return {
        "schema_version": "awm-wma-proposal-set-v1",
        "decision_id": "decision-01",
        "situation": {
            "remaining_h": 3.0,
            "incumbent": "models/current",
            "evidence": ["exp-01's permitted accuracy summary: 0.70"],
        },
        "scientist_preference": "A",
        "candidates": [
            {
                "candidate_id": candidate_id,
                "hypothesis": hypothesis,
                "parent_checkpoint": "models/current",
                "change": change,
                "train_h": train_h,
                "eval_h": 0.1,
                "cost_basis": "measured exp-01 throughput",
                "evidence": ["exp-01's permitted accuracy summary: 0.70"],
                "uncertainty": "new mixture's complete-run performance is unknown",
                "decision_test": "use the same allowed evaluator and compare the incumbent",
            }
            for candidate_id, hypothesis, change, train_h in (
                ("A", "more of the existing data might help", "repeat old mixture", 1.0),
                ("B", "new problem families might help", "add a new data mixture", 1.5),
            )
        ],
    }


@pytest.fixture
def context(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("State what the evidence tests.\n")
    private = tmp_path / "private"
    return session, skill, private


def comparison_output(proposal):
    ranking = [candidate["candidate_id"] for candidate in reversed(proposal["candidates"])]
    return {
        "schema_version": decisions.COMPARISON_SCHEMA,
        "decision_id": proposal["decision_id"],
        "proposal_sha256": decisions.proposal_sha(proposal),
        "ranking": ranking,
        "comparisons": [
            {
                "preferred": preferred,
                "alternative": alternative,
                "reason": "Prefer new data within the remaining measured budget.",
                "evidence": ["shared prior-run summary"],
                "uncertainty": "The new distribution has not yet been trained.",
            }
            for preferred, alternative in pairwise(ranking)
        ],
        "candidate_assessments": [
            {
                "candidate_id": candidate_id,
                "feasibility": "ready",
                "expected_effect": "Unknown endpoint; potentially useful improvement.",
                "opportunity_cost": "Uses budget that could test the competing mixture.",
                "uncertainty": "No full-run observation for this candidate.",
            }
            for candidate_id in ranking
        ],
        "suggestions": [],
    }


class RecordingBackend(backends.Backend):
    name = "recording"

    def __init__(self, callback=None):
        self.calls = []
        self.callback = callback

    def run(self, brief):
        self.calls.append(brief)
        proposal = json.loads(brief.card_path.read_text())
        output = comparison_output(proposal)
        # These copied placeholders must never be treated as measurements.
        output.update({
            "backend": "invented",
            "model": "invented",
            "effort": "invented",
            "wma_skill": "invented",
            "cost": {"usd": 9000, "gpu_min": 1000},
            "access": {"outside": ["invented"]},
            "leak_suspected": True,
        })
        if self.callback:
            self.callback(brief, output)
        brief.verdict_path.write_text(json.dumps(output))


def test_joint_comparison_is_one_call_over_frozen_candidates(context, proposal):
    session, skill, private = context
    before = copy.deepcopy(proposal)
    backend = RecordingBackend()

    result = compare.compare(
        session, proposal, backend,
        budget=backends.Budget(cpu_min=2, wall_min=3, max_turns=5),
        model="m-1", effort="high", skill_dir=skill, transcript_dir=private,
    )

    assert len(backend.calls) == 1
    brief = backend.calls[0]
    assert json.loads(brief.card_path.read_text()) == before == proposal
    assert brief.card_path.is_relative_to(private)
    assert brief.verdict_path.is_relative_to(private)
    assert not (session / "memory").exists()
    assert result["ranking"] == ["B", "A"]
    assert result["proposal_sha256"] == decisions.proposal_sha(before)
    assert result["backend"] == "recording" and result["model"] == "m-1"
    assert result["effort"] == "high" and result["wma_skill"] == schema.skill_sha(skill)
    assert set(result["cost"]) == {"wall_min"}
    assert result["access"] == {"status": "not_measured"}
    assert "leak_suspected" not in result
    assert brief.extra["output_kind"] == "comparison"
    assert brief.extra["proposal"] == before
    assert "ONE shared-context decision" in brief.prompt
    assert "not by\nconfidence" in brief.prompt
    assert "Do not read mutable live cards, future card results" in brief.prompt
    assert "separate launch review" in brief.prompt


def test_only_runtime_measurements_are_reported(context, proposal):
    session, skill, private = context

    def measured(brief, output):
        brief.extra["measured"] = {
            "model": "actual-model",
            "effort": "actual-effort",
            "cost": {"usd": 0.125, "turns": 2},
            "access": {"files": 2, "outside": ["/forbidden"]},
            "leak_suspected": True,
        }

    result = compare.compare(
        session, proposal, RecordingBackend(measured), model="requested-model",
        effort="requested-effort", skill_dir=skill, transcript_dir=private,
    )
    assert result["model"] == "actual-model" and result["effort"] == "actual-effort"
    assert result["cost"]["usd"] == 0.125 and result["cost"]["turns"] == 2
    assert result["cost"]["wall_min"] >= 0 and "gpu_min" not in result["cost"]
    assert result["access"] == {"files": 2, "outside": ["/forbidden"]}
    assert result["leak_suspected"] is True


def test_input_mutation_during_call_does_not_change_frozen_decision(context, proposal):
    session, skill, private = context
    digest = decisions.proposal_sha(proposal)

    def mutate_caller_input(brief, output):
        proposal["candidates"][0]["train_h"] = 900

    result = compare.compare(
        session, proposal, RecordingBackend(mutate_caller_input), skill_dir=skill,
        transcript_dir=private,
    )
    assert result["proposal_sha256"] == digest != decisions.proposal_sha(proposal)


def test_singleton_requires_a_reason_and_has_no_pairwise_comparison(context, proposal):
    session, skill, private = context
    proposal["candidates"] = proposal["candidates"][:1]
    backend = RecordingBackend()
    with pytest.raises(compare.CompareError, match="singleton"):
        compare.compare(session, proposal, backend, skill_dir=skill, transcript_dir=private)
    assert not backend.calls and not private.exists()

    proposal["singleton_reason"] = "Only this recipe has prepared data within the remaining hour."
    result = compare.compare(session, proposal, backend, skill_dir=skill, transcript_dir=private)
    assert len(backend.calls) == 1
    assert result["ranking"] == ["A"] and result["comparisons"] == []


def test_invalid_comparison_is_preserved_as_a_failed_attempt(context, proposal):
    session, skill, private = context

    def invalid(brief, output):
        output["ranking"] = ["B", "B"]

    backend = RecordingBackend(invalid)
    with pytest.raises(compare.CompareError, match="joint comparison failed"):
        compare.compare(session, proposal, backend, skill_dir=skill, transcript_dir=private)
    invocation = backend.calls[0].card_path.parent
    failure = json.loads((invocation / "failure.json").read_text())
    assert failure["proposal_sha256"] == decisions.proposal_sha(proposal)
    assert failure["cost"]["wall_min"] >= 0
    assert backend.calls[0].verdict_path.exists()


def test_snapshot_rewrite_is_detected_even_if_output_looks_valid(context, proposal):
    session, skill, private = context

    def rewrite_snapshot(brief, output):
        brief.card_path.chmod(0o644)
        brief.card_path.write_text("{}")

    with pytest.raises(compare.CompareError, match="modified the frozen proposal"):
        compare.compare(
            session, proposal, RecordingBackend(rewrite_snapshot), skill_dir=skill,
            transcript_dir=private,
        )


def test_backend_failure_keeps_partial_output_without_accepting_it(context, proposal):
    session, skill, private = context

    class Failing(RecordingBackend):
        def run(self, brief):
            super().run(brief)
            raise backends.BackendError("expired")

    backend = Failing()
    with pytest.raises(compare.CompareError, match="expired"):
        compare.compare(session, proposal, backend, skill_dir=skill, transcript_dir=private)
    invocation = backend.calls[0].card_path.parent
    assert (invocation / "failure.json").exists()
    assert backend.calls[0].verdict_path.exists()


@pytest.mark.parametrize("budget", [backends.Budget(wall_min=0), backends.Budget(max_turns=0)])
def test_comparison_requires_a_bounded_budget(context, proposal, budget):
    session, skill, private = context
    backend = RecordingBackend()
    with pytest.raises(compare.CompareError, match="budget"):
        compare.compare(session, proposal, backend, budget=budget, skill_dir=skill,
                        transcript_dir=private)
    assert not backend.calls


def test_single_card_heuristic_cannot_masquerade_as_joint_comparison(context, proposal):
    session, skill, private = context
    with pytest.raises(compare.CompareError, match="does not implement joint comparison"):
        compare.compare(session, proposal, backends.HeuristicBackend(), skill_dir=skill,
                        transcript_dir=private)
    assert not private.exists()
