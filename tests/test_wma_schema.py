"""A verdict is four levels with evidence; scoring it against the outcome is mechanical."""

from __future__ import annotations

import copy

import pytest

from awm.wma import schema
from exp_protocol_cards import closed_card, plan_card


def verdict() -> dict:
    return {
        "schema_version": schema.VERDICT_SCHEMA, "card_id": "exp-01", "wma_skill": "abc123abc123",
        "backend": "heuristic", "mode": "offline", "issued_at": "2026-09-02T00:00:00Z",
        "levels": {
            "L0_runs": {"answer": "yes", "confidence": 0.7, "basis": ["e1"]},
            "L1_valid": {"answer": "yes", "confidence": 0.6, "basis": []},
            "L2_effect": {"metric": "accuracy", "direction": "higher", "interval": [-0.02, 0.03],
                          "confidence": 0.4, "basis": []},
            "L3_worth_now": {"answer": "yes", "confidence": 0.5, "expected_cost_h": 1.5, "basis": []},
        },
        "evidence": [{"id": "e1", "path": "/t/memory/cards/exp-01.yaml", "locator": "setup", "note": "sft"}],
        "probes": [], "suggestions": {"preconditions": [], "cheaper_variants": []},
        "cost": {"cpu_min": 0, "gpu_min": 0, "wall_min": 0},
    }


class TestValidate:
    def test_example_is_valid(self) -> None:
        assert schema.validate_verdict(verdict()).ok

    @pytest.mark.parametrize("level", schema.LEVELS)
    def test_each_level_is_required(self, level: str) -> None:
        v = verdict()
        del v["levels"][level]
        assert any(p.field == f"levels.{level}" for p in schema.validate_verdict(v).errors)

    def test_bad_answer_bad_confidence_bad_interval(self) -> None:
        v = verdict()
        v["levels"]["L0_runs"]["answer"] = "maybe"
        v["levels"]["L1_valid"]["confidence"] = 1.5
        v["levels"]["L2_effect"]["interval"] = [0.05, -0.05]
        fields = {p.field for p in schema.validate_verdict(v).errors}
        assert {"levels.L0_runs.answer", "levels.L1_valid.confidence", "levels.L2_effect.interval"} <= fields

    def test_basis_must_name_an_evidence_id(self) -> None:
        v = verdict()
        v["levels"]["L0_runs"]["basis"] = ["e9"]
        assert any(p.field == "levels.L0_runs.basis" for p in schema.validate_verdict(v).errors)

    def test_round_trip_and_path(self, tmp_path) -> None:
        card = tmp_path / "memory" / "cards" / "exp-01.yaml"
        p = schema.verdict_path(card)
        assert p.name == "exp-01.verdict.json"
        schema.dump_verdict(p, verdict())
        assert schema.load_verdict(p) == verdict()

    def test_skill_sha_is_content_derived(self, tmp_path) -> None:
        (tmp_path / "SKILL.md").write_text("a")
        one = schema.skill_sha(tmp_path)
        (tmp_path / "SKILL.md").write_text("b")
        assert one != schema.skill_sha(tmp_path) and len(one) == 12


class TestTruth:
    def test_completed_card(self) -> None:
        t = schema.truth_from_card(closed_card())
        assert t["execution"] == "completed" and t["decision"] == "adopt" and t["output_checkpoint"]
        assert t["delta"] is None  # no comparator, no delta_vs_comparator

    def test_delta_from_field_then_from_comparator(self) -> None:
        card = closed_card()
        card["result"]["measurements"][0]["delta_vs_comparator"] = 0.05
        assert schema.truth_from_card(card)["delta"] == 0.05
        del card["result"]["measurements"][0]["delta_vs_comparator"]
        card["evaluation"]["comparator"] = {"ref": "base_model", "value": 0.33, "path": "/t/e.json"}
        assert schema.truth_from_card(card)["delta"] == pytest.approx(0.08)

    def test_open_card_has_no_truth(self) -> None:
        t = schema.truth_from_card(plan_card())
        assert t["execution"] is None and t["decision"] is None


class TestScore:
    def truth(self, **kw) -> dict:
        base = {"execution": "completed", "output_checkpoint": "/t/c", "measurements": [{"value": 0.4}],
                "decision": "adopt", "wall_h": 1.0, "delta": 0.02}
        base.update(kw)
        return base

    def test_all_hits(self) -> None:
        s = schema.score(verdict(), self.truth())
        assert s == {"L0": "hit", "L1": "hit", "L2": "in_interval", "L3": "hit"}

    def test_failed_run_misses_L0_and_L1(self) -> None:
        s = schema.score(verdict(), self.truth(execution="failed", output_checkpoint=None, measurements=[]))
        assert s["L0"] == "miss" and s["L1"] == "miss"

    def test_killed_counts_as_ran_but_not_valid(self) -> None:
        s = schema.score(verdict(), self.truth(execution="killed", output_checkpoint=None, measurements=[]))
        assert s["L0"] == "hit" and s["L1"] == "miss"

    def test_L2_above_below_unscorable(self) -> None:
        assert schema.score(verdict(), self.truth(delta=0.10))["L2"] == "above"
        assert schema.score(verdict(), self.truth(delta=-0.10))["L2"] == "below"
        assert schema.score(verdict(), self.truth(delta=None))["L2"] == "unscorable"

    def test_L3_no_and_defer_against_reject_and_adopt(self) -> None:
        v = verdict()
        v["levels"]["L3_worth_now"]["answer"] = "no"
        assert schema.score(v, self.truth(decision="reject"))["L3"] == "hit"
        assert schema.score(v, self.truth(decision="adopt"))["L3"] == "miss"
        v["levels"]["L3_worth_now"]["answer"] = "defer"
        assert schema.score(v, self.truth(decision="abandon_line"))["L3"] == "hit"
        assert schema.score(v, self.truth(decision="iterate"))["L3"] == "unscorable"

    def test_unknown_answers_are_unscorable(self) -> None:
        v = verdict()
        v["levels"]["L0_runs"]["answer"] = "unknown"
        assert schema.score(v, self.truth())["L0"] == "unscorable"

    def test_score_does_not_mutate(self) -> None:
        v = verdict()
        before = copy.deepcopy(v)
        schema.score(v, self.truth())
        assert v == before


def test_verdict_path_with_and_without_a_tag(tmp_path) -> None:
    card = tmp_path / "exp-01.yaml"
    assert schema.verdict_path(card).name == "exp-01.verdict.json"
    assert schema.verdict_path(card, tag="opus").name == "exp-01.verdict.opus.json"
    with pytest.raises(ValueError):
        schema.verdict_path(card, tag="bad tag!")
    assert schema.card_path_for(tmp_path / "exp-01.verdict.opus.json") == card
    assert schema.card_path_for(tmp_path / "exp-01.verdict.json") == card
