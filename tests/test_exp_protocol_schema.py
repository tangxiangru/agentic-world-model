"""Card v2: the pre-launch sections are checked before compute, the closing sections after."""

from __future__ import annotations

import pytest

from awm.exp_protocol import schema
from exp_protocol_cards import closed_card, plan_card


class TestPlan:
    def test_minimal_required_card_passes(self) -> None:
        assert schema.validate_plan(plan_card()).ok

    def test_wrong_schema_version_is_an_error(self) -> None:
        card = plan_card()
        card["schema_version"] = "awm-experiment-card-v1"
        r = schema.validate_plan(card)
        assert not r.ok and "schema_version" in r.errors[0].field

    @pytest.mark.parametrize("dotted", [
        "situation.trigger", "situation.elapsed_h", "problem.statement", "hypothesis.claim",
        "setup.parent_checkpoint.path", "setup.parent_checkpoint.origin", "setup.method.family",
        "setup.command.argv", "setup.command.cwd", "setup.output_dir", "setup.checkpoints.keep",
        "evaluation.protocol.n",
    ])
    def test_each_required_field_is_reported_by_name(self, dotted: str) -> None:
        card = plan_card()
        parent, _, leaf = dotted.rpartition(".")
        del schema.get(card, parent)[leaf]
        r = schema.validate_plan(card)
        assert not r.ok
        assert any(p.field == dotted for p in r.errors), [p.field for p in r.errors]

    def test_empty_data_list_is_an_error(self) -> None:
        card = plan_card()
        card["setup"]["data"] = []
        assert any(p.field == "setup.data" for p in schema.validate_plan(card).errors)

    def test_unknown_method_family_is_an_error(self) -> None:
        card = plan_card()
        card["setup"]["method"]["family"] = "alchemy"
        assert any(p.field == "setup.method.family" for p in schema.validate_plan(card).errors)

    def test_failed_contamination_check_is_an_error(self) -> None:
        card = plan_card()
        card["setup"]["data"][0]["contamination_check"] = "failed"
        assert any("contamination" in p.field for p in schema.validate_plan(card).errors)

    def test_score_target_as_claim_is_a_warning_not_an_error(self) -> None:
        card = plan_card()
        card["hypothesis"]["claim"] = "reach 85% on gsm8k"
        r = schema.validate_plan(card)
        assert r.ok and any(p.field == "hypothesis.claim" for p in r.warnings)

    def test_comparator_value_without_path_is_an_error(self) -> None:
        card = plan_card()
        card["evaluation"]["comparator"] = {"ref": "base_model", "value": 0.33}
        assert any(p.field == "evaluation.comparator.path" for p in schema.validate_plan(card).errors)

    def test_keep_accepts_policy_words_and_positive_ints_only(self) -> None:
        for good in ("all", "last", "best", 3):
            card = plan_card()
            card["setup"]["checkpoints"]["keep"] = good
            assert schema.validate_plan(card).ok, good
        card = plan_card()
        card["setup"]["checkpoints"]["keep"] = 0
        assert not schema.validate_plan(card).ok

    def test_paths_outside_the_session_dir_are_warnings(self, tmp_path) -> None:
        card = plan_card()
        r = schema.validate_plan(card, session_dir=tmp_path)
        assert r.ok
        assert any(p.field == "setup.output_dir" for p in r.warnings)


class TestResult:
    def test_closed_card_passes(self) -> None:
        assert schema.validate_result(closed_card()).ok

    def test_supported_without_measurement_is_an_error(self) -> None:
        card = closed_card()
        card["result"]["measurements"] = []
        assert any(p.field == "conclusion.verdict" for p in schema.validate_result(card).errors)

    def test_adopt_without_output_checkpoint_is_an_error(self) -> None:
        card = closed_card()
        card["result"]["output_checkpoint"] = None
        assert any(p.field == "result.output_checkpoint" for p in schema.validate_result(card).errors)

    def test_mechanism_verdict_needs_a_diagnostic_result(self) -> None:
        card = closed_card()
        card["conclusion"]["mechanism_verdict"] = "supported"
        assert any(p.field == "conclusion.mechanism_verdict" for p in schema.validate_result(card).errors)

    def test_missing_result_section_is_an_error(self) -> None:
        card = plan_card()
        r = schema.validate_result(card)
        assert any(p.field == "result" for p in r.errors)


class TestHashAndIo:
    def test_plan_hash_ignores_result_sections(self) -> None:
        a, b = plan_card(), closed_card()
        assert schema.plan_hash(a) == schema.plan_hash(b)

    def test_plan_hash_changes_when_a_plan_field_changes(self) -> None:
        a, b = plan_card(), plan_card()
        b["hypothesis"]["claim"] = "something else"
        assert schema.plan_hash(a) != schema.plan_hash(b)

    def test_round_trip_through_yaml(self, tmp_path) -> None:
        p = tmp_path / "exp-01.yaml"
        schema.dump_card(p, closed_card())
        assert schema.load_card(p) == closed_card()

    def test_minimal_card_has_every_plan_section_and_validates_only_after_filling(self) -> None:
        card = schema.minimal_card("exp-07")
        assert card["card_id"] == "exp-07"
        assert all(s in card for s in schema.PLAN_SECTIONS)
        assert not schema.validate_plan(card).ok
