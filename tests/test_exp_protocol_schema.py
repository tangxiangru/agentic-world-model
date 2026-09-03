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


# ---- review findings (2026-09-01) --------------------------------------------

class TestMeasurementTypes:
    def test_measurement_value_and_n_must_be_numbers(self) -> None:
        card = closed_card()
        card["result"]["measurements"][0]["value"] = "0.41"
        card["result"]["measurements"][0]["n"] = "one fifty"
        fields = {p.field for p in schema.validate_result(card).errors}
        assert {"result.measurements[0].value", "result.measurements[0].n"} <= fields


class TestV1Compat:
    def v1_card(self) -> dict:
        """A filled card in the six-section corpus format (jerry-dev card v1)."""
        return {
            "schema_version": "awm-experiment-card-v1", "card_id": "exp-02", "created_at": "2026-08-30T00:00:00Z",
            "elapsed_h": 2.5,
            "problem": {"statement": "slips", "evidence": [], "affected_share": None, "failure_examples": [],
                        "watch_set": None},
            "hypothesis": {"claim": "sft cuts slips", "mechanism": None,
                           "expected_effect": {"metric": "accuracy", "direction": "higher", "against": "base_model",
                                               "magnitude": None}, "falsified_if": None},
            "setup": {"parent_checkpoint": {"path": "google/gemma-3-4b-pt", "origin": "base_model", "hash": None},
                      "base_model": "google/gemma-3-4b-pt",
                      "data": [{"path": "/home/ben/task/d.jsonl", "source": "synthetic:self", "n_examples": 100,
                                "built_by": None, "build_command": [], "selection": "x",
                                "contamination_check": "passed", "mixture_weight": 1.0}],
                      "method": {"family": "sft", "framework": "trl", "peft": "none", "hyperparams": {"lr": 1e-5},
                                 "target_format": "x"},
                      "command": {"argv": ["python", "t.py"], "cwd": "/home/ben/task", "script": "/home/ben/task/t.py",
                                  "env": {}, "log": "/home/ben/task/l.log"},
                      "resume_argv": ["python", "t.py", "--resume_from_checkpoint", "{checkpoint}"],
                      "output_dir": "/home/ben/task/ckpts/exp-02",
                      "progress": {"unit": "optimizer_step", "total": 300},
                      "budget": {"gpu": "H100", "planned_h": 1.0}},
            "evaluation": {"protocol": {"command": [], "dev_set": "official --limit 150", "n": 150, "seed": 0},
                           "comparator": {"ref": "base_model", "value": 0.33, "path": "/home/ben/task/e.json"},
                           "diagnostic": {"what": None, "items": None, "n": None, "metric": "accuracy",
                                          "direction": "higher"}},
        }

    def test_migrate_v1_moves_elapsed_h_and_leaves_exactly_the_v2_gaps(self) -> None:
        card = schema.migrate_v1(self.v1_card())
        assert card["schema_version"] == schema.CARD_SCHEMA
        assert "elapsed_h" not in card and card["situation"]["elapsed_h"] == 2.5
        gaps = sorted(p.field for p in schema.validate_plan(card).errors)
        assert gaps == ["setup.checkpoints.keep", "setup.method.hyperparams.max_seq_len",
                        "setup.method.stop_token", "situation.trigger"]

    def test_migrate_is_a_no_op_on_a_v2_card(self) -> None:
        card = plan_card()
        assert schema.migrate_v1(card) == plan_card()


def test_render_says_ok_with_warnings_when_there_are_no_errors() -> None:
    """I10: the skill tells the scientist to loop until "ok"; warnings must not hide it."""
    r = schema.validate_plan(plan_card())
    assert r.ok and r.warnings
    text = r.render()
    assert text.splitlines()[-1].startswith("ok")


class TestTrainingFamiliesDeclareWhatPreflightNeeds:
    """#2 tightened: a card that trains must say what the eos and truncation checks need, or it cannot lock."""

    @pytest.mark.parametrize("family", schema.TRAINING_FAMILIES)
    def test_training_family_without_stop_token_and_max_seq_len_is_an_error(self, family: str) -> None:
        card = plan_card()
        card["setup"]["method"] = {"family": family}
        fields = {p.field for p in schema.validate_plan(card).errors}
        assert {"setup.method.stop_token", "setup.method.hyperparams.max_seq_len"} <= fields

    @pytest.mark.parametrize("family", ("merge", "decode-config", "other"))
    def test_non_training_family_needs_neither(self, family: str) -> None:
        card = plan_card()
        card["setup"]["method"] = {"family": family}
        assert schema.validate_plan(card).ok

    def test_max_seq_len_must_be_a_positive_int(self) -> None:
        card = plan_card()
        card["setup"]["method"]["hyperparams"]["max_seq_len"] = "2048"
        assert any(p.field == "setup.method.hyperparams.max_seq_len" for p in schema.validate_plan(card).errors)

    def test_answer_marker_stays_advisory(self) -> None:
        card = plan_card()
        card["setup"]["method"].pop("answer_marker", None)
        assert schema.validate_plan(card).ok
